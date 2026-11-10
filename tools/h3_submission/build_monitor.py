"""Bounded progress reporting for host builds; no model or device execution."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def tail(path, limit=4096):
    path = Path(path)
    if not path.is_file():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read(limit).decode('utf-8', errors='replace')


def run_logged(command, log, *, cwd=None, timeout=None, heartbeat=15,
               new_session=False, on_progress=None):
    """Return a report even on timeout. Caller decides whether to continue.

    Outer drivers create a process group; nested build commands inherit it so
    a timeout also terminates compiler descendants. Logs remain complete on disk.
    """
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = dict(command=list(map(str, command)), cwd=str(cwd) if cwd else None,
                  status='starting', elapsed_seconds=0, timed_out=False)
    process = None
    print('BUILD_START', log.name, flush=True)
    try:
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(report['command'], cwd=cwd, stdout=stream,
                                       stderr=subprocess.STDOUT,
                                       start_new_session=new_session)
            report.update(status='running', pid=process.pid)
            next_heartbeat = 0
            while True:
                elapsed = time.monotonic() - started
                report.update(elapsed_seconds=elapsed, log_bytes=log.stat().st_size)
                if elapsed >= next_heartbeat:
                    last = tail(log, 1024).splitlines()
                    report['last_line'] = last[-1][-240:] if last else ''
                    if on_progress:
                        on_progress(dict(report))
                    print('BUILD_PROGRESS', log.name, round(elapsed, 1),
                          report['log_bytes'], report['last_line'], flush=True)
                    next_heartbeat = elapsed + heartbeat
                code = process.poll()
                if code is not None:
                    report.update(status='passed' if code == 0 else 'failed', exit_code=code)
                    break
                if timeout is not None and elapsed >= timeout:
                    report.update(status='timed_out', timed_out=True)
                    if new_session and os.name == 'posix':
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        if new_session and os.name == 'posix':
                            os.killpg(process.pid, signal.SIGKILL)
                        else:
                            process.kill()
                        process.wait()
                    report.update(exit_code=124, terminated_returncode=process.returncode)
                    break
                time.sleep(min(0.2, heartbeat))
    except BaseException as error:
        if process is not None and process.poll() is None:
            if new_session and os.name == 'posix':
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
        report.update(status='interrupted', error=repr(error), exit_code=130)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        report['seconds'] = report['elapsed_seconds']
        report['tail'] = tail(log)
        if on_progress:
            on_progress(dict(report))
        print('BUILD_END', log.name, report['status'], report.get('exit_code'), flush=True)
    return report
