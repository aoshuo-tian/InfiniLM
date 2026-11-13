#!/usr/bin/env bash
# 仅在联网 Linux 准备机执行；不能在 Fabric 离线节点执行。
set -euo pipefail
TOOLS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$#" != 1 ]]; then
    printf '%s\n' '用法：bash prepare_offline.sh /新的/离线准备目录' >&2
    exit 2
fi
PREP="$1"
test ! -e "$PREP"
python3 -c 'import sys,platform,importlib.metadata as m; assert sys.platform=="linux" and sys.version_info[:2]==(3,11) and platform.machine()=="x86_64"; assert m.version("torch")=="2.9.0+das.opt1.dtk2604"; print("基础 NumPy",m.version("numpy"))'
mkdir -p "$PREP/logs"
PREP="$(cd -- "$PREP" && pwd)"
python3 -m venv --system-site-packages "$PREP/venv"
PY="$PREP/venv/bin/python"
"$PY" "$TOOLS/install_dependencies.py" --work "$PREP" --allow-online
"$PY" "$TOOLS/install_comparison.py" --work "$PREP" --allow-online
"$PY" "$TOOLS/install_dependencies.py" --work "$PREP" --check-only
"$PY" "$TOOLS/offline_wheels.py" --output "$PREP/offline" --download
printf '%s\n' '已准备 offline/wheels 与 manifest.json；厂商 Flash 另行复用或准备。请将离线目录上传评测者自己的 AFS。'
