"""Fabric租期与检查点编排；依赖、Flash、编译和推理均调用本仓库已有入口。"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import uuid

from build_checkpoint import pack_checkpoint, restore_checkpoint, sha
from build_monitor import run_logged, write_json

TOOLS = Path(__file__).resolve().parent


def load(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def configuration(base, cpu):
    path = base/'sessions/config.json'
    if path.exists():
        config = load(path)
    else:
        require(cpu, '先运行CPU构建')
        bundle = base/'assets/reviewer-public-sources.tar.gz'
        with tarfile.open(bundle) as archive:
            manifest = json.load(archive.extractfile('reviewer-source-manifest.json'))
        env = dict(os.environ,GIT_OPTIONAL_LOCKS='0')
        lm = TOOLS.parents[1]
        for name, repo in (('InfiniLM',lm),('InfiniCore',lm.parent/'InfiniCore')):
            head = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],env=env,text=True).strip()
            dirty = subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],env=env)
            require(head == manifest['commits'][name] and not dirty, '源码版本不匹配或checkout不干净：'+name)
        require(manifest.get('precompiled_framework_libraries') is False, '不得使用历史主机产物作为源码输入')
        config = dict(base=str(base),bundle=str(bundle),source_sha256=sha(bundle),entry_sha256=sha(__file__),
                      python_base=str(Path(sys.executable).resolve()),afs_group=base.stat().st_gid,
                      assets={key:os.environ[key] for key in ('H3_MODEL','H3_WHEELHOUSE','H3_FLASH_SO')})
        require(Path(config['assets']['H3_MODEL']).is_dir(), '模型目录不可读')
        require(Path(config['assets']['H3_WHEELHOUSE']).is_dir(), '先运行公开prepare_offline.sh准备依赖')
        require(Path(config['assets']['H3_FLASH_SO']).is_file(), '先运行公开prepare_flash.py准备Flash')
        write_json(path,config)
    require(config['base'] == str(base) and config['entry_sha256'] == sha(__file__), '入口或根目录已改变，拒绝混用')
    require(config['python_base'] == str(Path(sys.executable).resolve()), 'CPU/DCU基础Python路径不一致')
    require(sha(config['bundle']) == config['source_sha256'], '本次固定源码包SHA不匹配')
    return config


def evidence(work, run):
    if not work.exists():
        return
    for path in work.glob('*.json'):
        shutil.copyfile(path,run/path.name)
    for name in ('reviewer-base-constraints.txt','requirements-private-resolved.txt'):
        if (work/name).is_file():
            shutil.copyfile(work/name,run/name)
    if (work/'logs').is_dir():
        shutil.copytree(work/'logs',run/'public-logs',dirs_exist_ok=True)


def libraries(work):
    framework, ccl = load(work/'build-status.json'), load(work/'ccl-build-status.json')
    require(framework.get('passed') and framework.get('clean_host_framework') and ccl.get('passed')
            and ccl.get('complete'), '框架或通信库尚未成功')
    result = {**framework['libraries'],**ccl['libraries']}
    require(len(result) == 6, '最终动态库数量不是六项')
    for path, digest in result.items():
        require(Path(path).resolve().is_relative_to(work) and sha(path) == digest, '新库路径或SHA不匹配：'+path)
    return result


def session(args):
    import fcntl
    base = args.base.resolve()
    home = Path(os.environ['IFAB_AFS_HOME']).resolve()
    require(base != home and base.is_relative_to(home), '只使用评审自己的AFS子目录')
    (base/'sessions').mkdir(exist_ok=True)
    with (base/'sessions/lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        return execute(args,base)


def execute(args, base):
    cpu = args.command == 'cpu'
    config = configuration(base,cpu)
    latest_path = base/'sessions/latest-cpu.json'
    latest = load(latest_path) if latest_path.exists() else None
    require((not latest or latest['state'] == 'paused') if cpu else latest and latest['state'] == 'ready',
            'CPU只续本次paused记录；设备/比较阶段需CPU ready，失败不自动重试')
    if cpu:
        lease = dt.datetime.fromisoformat(args.lease_end)
        require(lease.tzinfo is not None, '到期时间必须含时区')
        remaining = lease.timestamp()-time.time()-args.reserve_seconds
        require(remaining >= 300, '剩余租期不足，不能开始本段')
        # 共享CPU单次租期有限，预算用尽先保存，只在下一会话续本次产物。
        # 单段最多1500秒，额外时间仍留给持久化；真正的到期预算优先。
        end = time.monotonic()+min(remaining,1500)
    else:
        end = time.monotonic()+args.timeout_seconds
    name = 'cpu' if cpu else args.stage
    marker = base/'sessions'/('stage-'+name+'.json')
    if not cpu:
        require(not marker.exists(), '已有该阶段记录，不覆盖、不自动重试')
        needs = {'communication':['verify'],'native':['verify','communication'],
                 'reference':['verify','communication'],'compare':['native','reference']}.get(name,[])
        require(all(load(base/'sessions'/('stage-'+key+'.json'))['passed'] for key in needs), '前置阶段未通过')
    run = base/'sessions'/(dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+name+'-'+uuid.uuid4().hex[:8])
    run.mkdir()
    require(shutil.disk_usage(args.scratch_parent).free >= 4*1024**3, '临时盘需至少4GiB可用空间')
    root = Path(tempfile.mkdtemp(prefix='h3-fabric-',dir=args.scratch_parent)).resolve()/'repro'
    work = root/'build'
    report = dict(passed=False,complete=False,root=str(root),source_sha256=config['source_sha256'],stages=[])
    if not cpu:
        write_json(marker,dict(passed=False,complete=False,run=str(run)))
    os.environ.update(config['assets'],H3_REPRO_ROOT=str(root),H3_OUTPUT_ROOT=str(base/'results'))
    for key in ('H3_FLASH_WHEEL','H3_REFERENCE_VIDEO','H3_NATIVE_VIDEO','FA_SO_PATH','FLASH_ATTN_PREBUILT'):
        os.environ.pop(key,None)
    active_tools = TOOLS

    def direct(script, *arguments):
        # 直接调用公开Python CLI，避免嵌套GNU timeout另建进程组，确保预算到期能终止全部编译子进程。
        return ['bash','-c','source "$1"; shift; exec python "$@"','h3-stage',str(work/'setup.sh'),
                str(active_tools/script),*map(str,arguments)]

    def stage(label, command=None, resumable=False):
        timeout = end-time.monotonic()
        require(timeout > 5, '预算不足以启动阶段：'+label)
        started = time.time()
        def progress(row):
            report['active'] = dict(stage=label,**row)
            write_json(run/'execution-record.json',report)
        row = run_logged(command or ['bash',str(active_tools/'reviewer_reproduce.sh'),label],run/(label+'.log'),timeout=timeout,
                         new_session=True,on_progress=progress)
        report['stages'].append(dict(stage=label,**row))
        if resumable and row.get('timed_out'):
            return False
        if row['exit_code']:
            logs = sorted((p for p in (work/'logs').glob('*.log') if p.stat().st_mtime >= started),key=lambda p:p.stat().st_mtime)
            logs.append(run/(label+'.log'))
            for log in logs:
                lines = log.read_text(errors='replace').splitlines()
                found = next((i for i,line in enumerate(lines) if re.search(r'Traceback|fatal error:|cannot find -l|undefined reference|\berror:',line,re.I)),None)
                if found is not None:
                    report['first_error'] = dict(log=str(log),line=found+1,text='\n'.join(lines[found:found+35]))
                    break
            raise RuntimeError('阶段失败，停止：'+label)
        return True

    try:
        if latest:
            cp = latest['checkpoint']
            require(latest['source_sha256'] == config['source_sha256']
                    and Path(cp['path']).resolve().is_relative_to(base/'sessions'), '禁止恢复其他实验检查点')
            write_json(run/'restore.json',restore_checkpoint(cp['path'],cp['sha256'],config['source_sha256'],root))
            stage('private-venv',[sys.executable,'-m','venv','--system-site-packages','--without-pip',str(work/'venv')])
        else:
            stage('prepare')
        active_tools = work/'sources/InfiniLM/tools/h3_submission'
        for key in ('dependencies','comparison-dependencies','flash'):
            stage(key)
        if cpu:
            state = 'ready'
            for key in ('framework','ccl','native-extension-import'):
                if end-time.monotonic() < 90:
                    state = 'paused'
                    break
                if key == 'ccl' and (work/'ccl-build-status.json').exists() and load(work/'ccl-build-status.json').get('passed'):
                    libraries(work)
                    continue
                command = None
                if key == 'framework':
                    status = load(work/'build-status.json') if (work/'build-status.json').exists() else {}
                    option = ['--verify-completed'] if status.get('passed') else (['--resume'] if (work/'build-origin.json').exists() else [])
                    command = direct('build_framework.py',*option)
                elif key == 'ccl':
                    command = direct('build_ccl.py')
                if key == 'native-extension-import':
                    command = direct('runtime_links.py','--verify-native',work)
                if not stage(key,command,resumable=key in ('framework','ccl')):
                    state = 'paused'
                    break
            require(load(work/'build-origin.json').get('initial_clean_host_framework') is True, '缺少干净起步证据')
            if state == 'ready':
                report['libraries'] = libraries(work)
            evidence(work,run)
            require(shutil.disk_usage(base).free >= (args.checkpoint_mib+64)*1024**2, '持久空间不足以保存检查点')
            cp = pack_checkpoint(root,run/'checkpoint.tar.gz',config['source_sha256'],maximum=args.checkpoint_mib*1024**2)
            archive = Path(cp['path'])
            require(archive.stat().st_gid == config['afs_group'], '检查点组与当前AFS组不一致，不自动改组')
            archive.chmod(0o640)
            require(sha(archive) == cp['sha256'], '保存后读回SHA失败')
            report.update(state=state,passed=state=='ready',checkpoint=cp)
            write_json(latest_path,dict(state=state,source_sha256=config['source_sha256'],checkpoint=cp,run=str(run)))
            return 0 if state=='ready' else 75
        command = None
        if args.stage == 'communication':
            (base/'results').mkdir(exist_ok=True)
            command = direct('check_rebuilt_ccl.py','--work',work,'--output',base/'results/communication')
        elif args.stage in ('verify','native','reference'):
            flash = load(work/'vendor-flash/prepare.json')
            image = active_tools.parents[1]/'examples/h3_submission/tp8_20261109/profile-sources/user-puppy-clear-8dacfbb2.jpg'
            output = base/'results'/('preflight' if args.stage == 'verify' else args.stage)
            option = ['--verify-only'] if args.stage == 'verify' else ['--backend',args.stage]
            command = direct('run_rebuilt.py','--work',work,'--model',config['assets']['H3_MODEL'],
                             '--first-frame',image,'--flash-so',flash['flash_so'],
                             '--flash-sha256',flash['flash_so_sha256'],'--output',output,*option)
        stage(args.stage,command)
        report['libraries'] = libraries(work)
        report['passed'] = True
        return 0
    except BaseException:
        report.update(error=traceback.format_exc(),passed=False)
        if cpu:
            write_json(latest_path,dict(state='failed',run=str(run),source_sha256=config['source_sha256']))
        raise
    finally:
        try:
            evidence(work,run)
            report['complete'] = True
            write_json(run/'execution-record.json',report)
            if not cpu:
                write_json(marker,dict(passed=report['passed'],complete=True,run=str(run)))
            elif 'error' not in report and report.get('state') in ('ready','paused'):
                print('CPU_READY' if report['state']=='ready' else 'CPU_PAUSED_CHECKPOINT_VERIFIED_EXIT_75',flush=True)
        except BaseException:
            if cpu:
                write_json(latest_path,dict(state='failed',run=str(run),source_sha256=config['source_sha256']))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    for mode in ('cpu','run'):
        p = sub.add_parser(mode)
        p.add_argument('--base',type=Path,required=True)
        p.add_argument('--scratch-parent',type=Path,default=Path(tempfile.gettempdir()))
        if mode == 'cpu':
            p.add_argument('--lease-end',required=True)
            p.add_argument('--reserve-seconds',type=int,default=180)
            p.add_argument('--checkpoint-mib',type=int,default=512)
        else:
            p.add_argument('--stage',required=True,choices=('verify','communication','native','reference','compare'))
            p.add_argument('--timeout-seconds',type=int,default=1740)
    args = parser.parse_args()
    require(sys.platform == 'linux' and sys.version_info[:2] == (3,11), '需要指定Linux/Python3.11镜像')
    if args.command == 'cpu':
        require(120 <= args.reserve_seconds <= 600 and 256 <= args.checkpoint_mib <= 1024, '保存预算或归档上限不合适')
    else:
        require(60 <= args.timeout_seconds <= 1740, '程序预算需在60–1740秒以内')
    os.umask(0o007)
    return session(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
