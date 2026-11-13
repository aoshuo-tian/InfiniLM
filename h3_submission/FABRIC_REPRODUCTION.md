# Fabric 公开源码复现入口（2026-11-13 修订）

本文件说明当前交付代码的使用方式。它补齐构建接线与离线依赖准备，**没有在本次不可用的 DCU 上重新完成全链路验收**，也不对应一个预先保证的 SSIM 分数。历史日志与得分不能改称本次提交的新测试结果。

## 运行边界

- 环境：Linux x86_64、Python 3.11、DTK 26.04、已有海光 Torch `2.9.0+das.opt1.dtk2604`。复用镜像的 NumPy；最后一次实际运行记录为 `1.25.0`。CPU 编译环境与 DCU 运行环境的基础包身份必须一致。
- 代码：报告固定三个公开仓库的完整 commit。Core 分支 `minimax-h3`；LM 与 NineToothed 分支 `minimax_h3`。仓库与上游/依赖 commit 见同目录 `dependencies.json`。
- 模型、海光 Torch/驱动/DTK、厂商 Flash 是外部输入，不以作者旧 AFS 路径或旧运行包代替。不要将这些大文件放进 Fabric 项目启动文件。
- 构建复用提交中的 gfx936 内嵌设备代码，重新编译主机包装、Core、Python 扩展、LM 与 CCL；不声称本流程重新调用九齿生成全部设备代码。
- QKNorm/QKRoPE 静态库现在由 xmake 的 `h3-qknorm-aot` / `h3-qkrope-aot` 目标从公开源码构建。`_infinilm` 自动依赖它们，不要求评审预先拥有 `.a` 文件。QKRoPE 的实验开关仍关闭；构建支持库不等于启用该候选。

## 一、在联网准备机获取并传输公开源码

由报告提供 `CORE_COMMIT`、`LM_COMMIT`、`NINE_COMMIT` 的完整 40 位值。不要用一个尚未发布的本地版本替代。下面代码在**联网准备机**执行，不在 Fabric 离线共享 CPU 内执行：

```bash
set -euo pipefail
: "${CORE_COMMIT:?从报告复制完整 Core commit}"
: "${LM_COMMIT:?从报告复制完整 LM commit}"
: "${NINE_COMMIT:?从报告复制完整 NineToothed commit}"
PREP="$PWD/h3-public-inputs"
test ! -e "$PREP"
mkdir -p "$PREP/code"
git -c core.autocrlf=false clone --branch minimax-h3 https://github.com/aoshuo-tian/InfiniCore.git "$PREP/code/InfiniCore"
git -c core.autocrlf=false clone --branch minimax_h3 https://github.com/aoshuo-tian/InfiniLM.git "$PREP/code/InfiniLM"
git -c core.autocrlf=false clone --branch minimax_h3 https://github.com/aoshuo-tian/ninetoothed.git "$PREP/code/ninetoothed"
git -C "$PREP/code/InfiniCore" checkout --detach "$CORE_COMMIT"
git -C "$PREP/code/InfiniLM" checkout --detach "$LM_COMMIT"
git -C "$PREP/code/ninetoothed" checkout --detach "$NINE_COMMIT"
git -C "$PREP/code/InfiniCore" submodule update --init --recursive
git -C "$PREP/code/InfiniLM" submodule update --init --recursive
python3 - "$PREP" "$CORE_COMMIT" "$LM_COMMIT" "$NINE_COMMIT" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]); pins=dict(zip(('InfiniCore','InfiniLM','ninetoothed'),sys.argv[2:]))
files={}
for name,pin in pins.items():
    repo=root/'code'/name
    assert len(pin)==40 and subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==pin
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no']).strip()
    for p in repo.rglob('*'):
        if not p.is_file() or '.git' in p.relative_to(root).parts: continue
        if p.suffix in ('.o','.a','.so','.pyc'): raise RuntimeError('源码树混入编译产物：'+str(p))
        files[p.relative_to(root).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
(root/'source-identity.json').write_text(json.dumps(dict(commits=pins,files=files),indent=2))
PY
tar --exclude='.git' -czf "$PREP/h3-source-tree.tar.gz" -C "$PREP" code source-identity.json
sha256sum "$PREP/h3-source-tree.tar.gz"
```

源码包只是无法联网时的传输方式，提交凭据仍为报告中的公开仓库、分支和 commit。保存打印的 SHA，通过自己的传输渠道核对。

## 二、先检查复用，确实缺包时再准备离线 wheel

`dependencies` 与 `comparison-dependencies` 先在实际 build venv 检查版本、导入及 H3 接口；已有兼容包就直接复用，不调用 pip。缺少/版本不符才要求离线 wheelhouse。包版本符合但导入/API 损坏时停止，保留诊断，不盲目重装。默认不联网。

若 Fabric 基础镜像缺依赖，可在**匹配目标基础镜像的联网 Linux x86_64 Python3.11 准备机**执行：

```bash
bash "$PREP/code/InfiniLM/tools/h3_submission/prepare_offline.sh" "$PREP/python-preparation"
```

此命令创建专用 venv，保护基础 Torch/NumPy，检查 H3 与评分接口，再收集实际已解析的传递依赖闭包。输出 `python-preparation/offline/wheels/`、`requirements-resolved.txt`、带 wheel SHA 的 `manifest.json`。不是下载 Windows 包；不是把 wheel 全部塞进 GitHub。若准备机没有匹配的厂商 Torch/基础镜像，该步骤不能代替兼容性验证。联网下载尚需软件源实际可访问；本次未在目标 Linux 上重跑下载。

Flash 优先复用评测环境中**身份相同**的现有动态库：设置 `H3_FLASH_SO`，或由当前解释器包目录/显式 `FA_SO_PATH`、`FLASH_ATTN_PREBUILT` 发现。不会遍历作者 AFS。文件需为 ELF64 x86_64、2088777840 字节、SHA256：

`a56d9b54127d86777b55b41e06b55d246897f7ea202048af8413d9d65f1f4395`

没有此库才另行提供完整厂商 wheel：

`flash_attn-2.6.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl`

658898543 字节，SHA256：

`8546e46bcbfcd346757c67a0ff180ace2a49716d9724d52ee870d04b4c3b41d1`

准备脚本只校验并提取 `.so`，不 pip 安装 Flash wheel；文件 SHA 通过不代表 DCU 执行已通过。后续仍保留强制 Flash 的设备门禁，不回退 MATH。

## 三、在 Fabric 自己的持久空间准备输入

通过网页上传到评测者**自己的个人 AFS**：`h3-source-tree.tar.gz`、如需则上传离线 `wheels/` 与 `manifest.json`、如需则上传厂商 Flash wheel。请放在 `$IFAB_AFS_HOME/h3-review/inputs`，不要放进项目启动文件，不引用作者私人目录。下面在共享 CPU 终端执行：

```bash
set -euo pipefail
: "${IFAB_AFS_HOME:?需要 Fabric 个人 AFS 根目录}"
BASE="$IFAB_AFS_HOME/h3-review"
INPUTS="$BASE/inputs"
test ! -e "$BASE/code"
# 先将此处与准备机打印的源码包 SHA 进行核对，再解包。
sha256sum "$INPUTS/h3-source-tree.tar.gz"
tar -xzf "$INPUTS/h3-source-tree.tar.gz" -C "$BASE"
python3 - "$BASE" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); manifest=json.loads((root/'source-identity.json').read_text())
for rel,digest in manifest['files'].items():
    path=(root/rel).resolve()
    assert path.is_relative_to(root.resolve())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,rel
print('源码文件身份核对通过',manifest['commits'])
PY
export H3_REPRO_ROOT="$BASE/repro"
export H3_MODEL="/public/home/aclsylqidf/shared/MiniMax-H3/diffusers"
# 若已上传离线 wheel 目录，设置；已有依赖通过检查时不会安装。
if [[ -d "$INPUTS/wheels" ]]; then export H3_WHEELHOUSE="$INPUTS/wheels"; fi
# 可改为评测环境中真实存在、身份匹配的 .so 路径。
# export H3_FLASH_SO="/实际持久目录/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so"
WHEEL="$INPUTS/flash_attn-2.6.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl"
if [[ -f "$WHEEL" ]]; then export H3_FLASH_WHEEL="$WHEEL"; fi
declare -p H3_REPRO_ROOT H3_MODEL > "$BASE/fabric-env.sh"
for NAME in H3_WHEELHOUSE H3_FLASH_SO H3_FLASH_WHEEL; do
    if [[ -v "$NAME" ]]; then declare -p "$NAME" >> "$BASE/fabric-env.sh"; fi
done
ENTRY="$BASE/code/InfiniLM/tools/h3_submission/reviewer_reproduce.sh"
bash "$ENTRY" prepare
bash "$ENTRY" dependencies
bash "$ENTRY" comparison-dependencies
bash "$ENTRY" flash
```

`prepare` 会先检查构建文件系统能执行文件且支持相对符号链接；失败就停止，不能继续把 AFS 当作可直接编译/运行的 POSIX 目录。另需预留足够编译空间，网页额度与文件系统剩余空间不是同一个指标。此持久根目录方案仍需在实际节点验证；历史运行使用过临时工作区与检查点，不能把它称为已验证的同一存储流程。

若基础包或 Flash 检查失败，先查看 `build/*probe.json` / 错误提示，不申请 DCU 来重复检查下载。`prepare` 对现有目录拒绝覆盖；恢复编译时不要重复运行它。

## 四、共享 CPU 分阶段编译

新 CPU 会话先恢复环境，然后一次运行一个阶段。超时后保留同一持久目录，再执行相同的 `framework`；脚本验证源码指纹并复用对象，不自动申请新会话。不要在旧断点目录中混入新版源码。

```bash
set -euo pipefail
BASE="$IFAB_AFS_HOME/h3-review"
source "$BASE/fabric-env.sh"
ENTRY="$BASE/code/InfiniLM/tools/h3_submission/reviewer_reproduce.sh"
bash "$ENTRY" framework
bash "$ENTRY" ccl
```

`framework` 构建 Core 主体、Core Python 扩展、QK 支持库和 LM Python 扩展；`ccl` 构建通信库。阶段日志在 `$H3_REPRO_ROOT/build/logs`，状态与源码指纹在同一 build 目录。CPU 只做主机编译与导入，不把它描述为 DCU 内核或视频测试。

## 五、分配八卡后核对实际新库和通信

以下需要可用的八卡 gfx936 DCU 资源；本脚本不自动申请/续期资源。CPU 与 DCU 基础包不匹配时停止，不将普通 Torch 替换海光 Torch。

```bash
set -euo pipefail
BASE="$IFAB_AFS_HOME/h3-review"
source "$BASE/fabric-env.sh"
ENTRY="$BASE/code/InfiniLM/tools/h3_submission/reviewer_reproduce.sh"
bash "$ENTRY" verify
bash "$ENTRY" communication
```

设备门禁记录实际加载库位置、SHA、厂商 Flash 的 DCU 执行结果。通信阶段直接验证 2/4/8 卡。所有检查通过后才进行生成。

## 六、使用新编译库生成原生视频

```bash
bash "$ENTRY" native
```

入口冻结：同一清晰小狗图、`A dog falls asleep`、seed42、宽768×高928、50请求步、124帧、24FPS；两个原生引擎各自在同组0–7卡做TP8。VAE/cast/混合精度/厂商Flash与选定基线一致，不包含后续九齿 VAE/SwiGLU/GC 等候选。此阶段单独预留充足资源时间；失败不自动重试。输出在 `$H3_REPRO_ROOT/results/native`。

## 七、PyTorch 参照及原 compare 评分

参照生成也需要八卡资源，使用官方 legacy 路径：text0–3、Transformer4–7的模块分布，不称为PyTorch TP8。基础参数与原生视频一致。可在另一会话执行；先恢复上面的 BASE、环境与 ENTRY。

```bash
bash "$ENTRY" reference
```

两端成功后在 CPU 会话运行：

```bash
set -euo pipefail
BASE="$IFAB_AFS_HOME/h3-review"
source "$BASE/fabric-env.sh"
ENTRY="$BASE/code/InfiniLM/tools/h3_submission/reviewer_reproduce.sh"
bash "$ENTRY" comparison-dependencies
bash "$ENTRY" compare
```

`compare` 调用未修改的 `examples/minimax_h3_compare.py`，传入已有两段视频，不再生成；输出尺寸、帧数检查及平均 SSIM，结果位于 `results/compare` 与 `results/compare.stdout.log`。它不预填历史82.468%或其他分数。

若复用已有 PyTorch 参照，可显式设置 `H3_REFERENCE_VIDEO`，但必须另附对应权重、输入图、提示词、seed、分辨率、帧数和步数的记录，不能只按文件名假定一致。原生/参照/评分输出拒绝覆盖，改 `H3_OUTPUT_ROOT` 开新结果目录。

## 尚需实际目标验证

主机单元测试、源文件/头文件解析、Windows xmake 目标图检查，不能替代 Linux DTK 目标编译、Flash 动态加载、多卡通信或视频质量验收。此次修订未产生新视频或得分；评审运行遇到首个错误时应保留阶段状态与完整日志，不能据此直接改用作者旧二进制包。
