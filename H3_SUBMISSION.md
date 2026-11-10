# MiniMax-H3：shared TP8 cache 交付版本

本次提交对应 2026-11-09 已实际执行的 FL2VA TP8 cache 基线，不包含随后的视频 VAE 九齿 Attention、SwiGLU、cast 或 GC 实验候选。它不是完全未经修改的上游版本：保留了此前已采用的 InfiniCore/InfiniLM 算子适配、source-device 输出桥接和有界 KV 分配策略。

## 代码与分支

| 工程 | 仓库 | 交付分支 | 上游基点 |
|---|---|---|---|
| InfiniCore | https://github.com/aoshuo-tian/InfiniCore | `minimax-h3` | `f32e2fbb2a9b2ca912b5711f4f3836c403196a74` |
| InfiniLM | https://github.com/aoshuo-tian/InfiniLM | `minimax_h3` | `65deb91f840ebaef752d028fad2f3217ce0950f4` |

两套源码取自已验证运行包 `tp-qualified-runtime.tar.gz`，SHA256 为 `3e79d685a81ff92db2ef24282b8252e9839807f8706250713d9a0cf98db81450`。未上传整个运行环境。`h3_submission/runtime-source-manifest.json` 列出从实际运行包复制的修改文件和生成源码；其他上游文件保留，比较时仅忽略 CRLF/LF 差异。

## 已执行的配置与结果

| 项目 | 已测试配置 |
|---|---|
| 提示词 | `A dog falls asleep` |
| 输入照片 | `examples/h3_submission/tp8_20261109/profile-sources/user-puppy-clear-8dacfbb2.jpg` |
| 图片 SHA256 | `8dacfbb2bf2ca5593c8b33222b74e7c265380c8441a41bc09718083fe1ca3fd8` |
| 宽 × 高 | **768 × 928** |
| 帧数、帧率 | 124 帧、24 FPS；视频约 5.17 秒 |
| 步数、种子 | 请求 50 步，实际 49 次 Transformer 调用；seed 42 |
| 原生条件编码器 | InfiniLM，TP 0–7 |
| 原生 H3 Transformer | InfiniLM/InfiniCore，TP 0–7；与条件编码器使用同一组卡 |
| PyTorch 参照 | legacy；text 允许使用 0–3、Transformer 4–7，通过 device_map 分布模块，不能称为 TP8 |
| 视频 VAE、音频 VAE、scheduler | 既有 Torch/Diffusers 路径；保留 BF16 VAE 权重、既有 cast/混合精度及厂商 Flash |

```text
[result] MiniMax-H3 video comparison
  FL2VA
    Size:         PASS (PyTorch 768x928, InfiniLM 768x928)
    Length:       PASS (PyTorch 124 frames, InfiniLM 124 frames)
    Average SSIM: 82.468%
```

原生生成与比较程序均退出 0，资源已释放。该分数是一个样例的逐帧平均 SSIM，不代表已通过某个未给定的正式质量阈值，也不是性能对照。此批视频尚未独立进行完整视觉验收。`h3_submission/tested-result.json` 记录输出身份；没有把旧 10 步或 4+4 视频当成这次结果。

## 入口及使用范围

精确保留的平台入口：

```bash
python examples/h3_submission/tp8_20261109/h3_run_tp8_768x928_s50.py
```

**无参数只打印计划，不加载模型、不申请资源。** 实际平台运行使用同一入口的 `--platform` 参数，由平台提供资源和 `IFAB_AFS_HOME`。源码目录中保留了原始 ZIP、解开的九个父脚本及 profile/helper，SHA 门禁保留；不要只挑其中一个旧父脚本直接运行，它可能包含历史默认参数。最终采用的是外层 TP8 profile。

该精确入口依赖原平台已经存在的运行包、私有离线依赖和厂商 Flash 缓存，AFS 相对位置写在父脚本里。它是**已运行配置的保留件，不是下载本仓库就能自动恢复全部外部资产的通用安装器**。模型权重、厂商二进制和整个 Python 环境不在 Git 仓库中。移到新账户时应准备这些资产并明确调整路径；不能声称原始文件 SHA 未变。

关键文件身份：

| 文件 | SHA256 |
|---|---|
| `examples/minimax_h3.py` | `ca54a18a73a7d760d644ca4eb2c3b20678f88dfdbcfa26a62dcd3de617b31af4` |
| `python/infinilm/multimodal/minimax_h3.py` | `5e5fc07bf0c02925c553603285fd163c44c973eb6c3de71349cf6c7677e3e4e5` |
| `examples/minimax_h3_compare.py` | `e143c28e4b3acb885662e0aa238aa00ca17cbcc82077f465f2ce57d0f1225590` |

`.gitattributes` 保留这些文件和冻结 profile 的原始字节，包括换行，以免跨平台 checkout 破坏 SHA 校验。

## 从源码构建

构建环境是 Linux x86_64、海光 gfx936、DTK 26.04、Python 3.11 和系统海光 PyTorch `2.9.0+das.opt1.dtk2604`，xmake 3.0.5；需要 GCC/G++、ar、Python 开发头、RCCL 兼容的 `nccl.h`/`libnccl`。不要在本地 Windows 安装或替换海光 Torch。

```bash
git clone --branch minimax-h3 https://github.com/aoshuo-tian/InfiniCore.git
git clone --branch minimax_h3 https://github.com/aoshuo-tian/InfiniLM.git
git -C InfiniCore submodule update --init third_party/spdlog
git -C InfiniLM submodule update --init third_party/spdlog third_party/json

# 新工作目录必须尚不存在。先查看计划，准备阶段也不会申请 GPU。
python InfiniLM/tools/h3_submission/prepare_build.py --core InfiniCore --work "$PWD/h3-build" --plan
python InfiniLM/tools/h3_submission/prepare_build.py --core InfiniCore --work "$PWD/h3-build"
python InfiniLM/tools/h3_submission/install_dependencies.py --work "$PWD/h3-build"
source h3-build/setup.sh
python InfiniLM/tools/h3_submission/build_framework.py

# 补启通信后才能做真实 TP；不能拿 --ccl=n 的基础构建当作 TP 就绪。
python InfiniLM/tools/h3_submission/build_ccl.py --plan
python InfiniLM/tools/h3_submission/build_ccl.py
```

构建会在新目录编译主机框架、Python 扩展、QK 归一化等 AOT 包装。已有 gfx936 设备代码以生成 C/C++ 中的嵌入数组保留，**本步骤复用它们，不重新生成全部设备内核，也不要求仅为本构建安装 Triton**。`build/ninetoothed*` 在此版本中包含生成源码，不能把这些目录当成普通输出全部删掉。

证据边界：`build_framework.py` 保留实际运行包中的历史构建脚本，基础构建使用 `--ccl=n`。2026-11-10 已取回历史成功的 CCL-on 构建记录和测试脚本，确认仅将同组配置的 `--ccl=n` 改为 `--ccl=y`，再构建 `infiniccl` 目标。其库 SHA 与已测试运行包一致。`h3_submission/ccl-build-provenance.json` 保留命令及原件身份，补建脚本会核对全部配置。**这些是历史证据；新整理的构建、依赖准备、产物迁移及生成入口仍待目标环境实跑，不能当作已完成清洁复现。**

运行视频前仍需实际安装/导入相容的私有 Python 依赖。已用环境包括 Transformers 5.10.0、Diffusers 0.40.0、Accelerate 1.15.0、PyAV 18.1.0、torchvision 0.24.0+cpu、janus 2.0.0、xxhash 3.2.0，NumPy 和海光 Torch 沿用系统版本。这不是让所有环境盲目重装的列表；先检查实际推理解释器。

厂商 Flash wheel：`flash_attn-2.6.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl`，SHA256 `8546e46bcbfcd346757c67a0ff180ace2a49716d9724d52ee870d04b4c3b41d1`。按厂商渠道准备动态库并设置 `FA_SO_PATH`，保留强制 Flash 检查，不自动回退到 MATH。该依赖不是本队实现的九齿算子。

## 复用已有视频评分

评分需要可导入 PyAV、scikit-image 及其 SciPy 依赖的 CPU Python；不需要重新生成视频：

```bash
python examples/minimax_h3_compare.py --tasks fl2va \
  --reference-fl2va-video /path/to/reference-768x928-s50.mp4 \
  --infinilm-fl2va-video /path/to/native-tp8-768x928-s50.mp4 \
  --output-dir /path/to/compare-output
```

以上两个视频均提供时会跳过生成。单纯用原 compare 的生成入口，不会自动应用这里冻结的 cache/profile；要复现 82.468% 必须使用本次选定的生成版本与同一输入。

## 第三方与限制

保留两工程原许可证和子模块引用；构建支持中的 CCCL、pybind11、OpenMP 头及许可证单独保留。AOT 生成件含设备代码，适用架构固定 gfx936，不能宣称为任意 DCU 通用包。模型及厂商二进制应遵守各自许可证。

后续 VAE 替换、视觉 RoPE/MHA 候选、分片 GC 等其他实验没有合并到这个选定运行版本。公开整理没有额外占用平台卡时；此提交不新增画质修复、正式评分通过或端到端加速结论。

配套 Core 精确 commit：`70c7049340e18d5c60f6f06f6dc213f4a1d7bf39`；机器可读固定信息见 `h3_submission/dependencies.json`。另附 `tools/h3_submission/kernel_sources/` 中的 QK NineToothed 图表达与历史生成器，从已校验的局部验证归档取出；QK+RoPE 仍关闭。这些是源码来源补充，不宣称本轮重新生成设备代码或重放历史调优。


## 2026-11-10 独立评审复现入口（待目标验证）

新入口 `tools/h3_submission/run_rebuilt.py` 只接受成功的新建框架及 CCL 构建记录。运行前实际核查两个 Python 包和六类已加载动态库的路径、SHA，拒绝混入旧框架库；不会调用历史平台入口恢复旧 runtime。无参数或 `--plan` 只核对计划，不导入 Torch 或加载模型。

与冻结 TP8 worker 相比，只把输入图片、Flash 库/描述文件和输出目录改为显式参数；生成计算、TP8、缓存策略及原 VAE 精度/cast 保持不变。`host_checks.py` 包含反向恢复后的完整 worker AST 比较。

```bash
# 只在已完成前述新建构建、并 source setup.sh 后执行。
# MODEL、FLASH_SO 由评审环境提供；库 SHA 应来自厂商包校验记录。
python InfiniLM/tools/h3_submission/run_rebuilt.py --plan
python InfiniLM/tools/h3_submission/run_rebuilt.py \
  --work "$PWD/h3-build" --model "$MODEL" --output "$PWD/h3-new-video" \
  --first-frame InfiniLM/examples/h3_submission/tp8_20261109/profile-sources/user-puppy-clear-8dacfbb2.jpg \
  --flash-so "$FLASH_SO" --flash-sha256 "$FLASH_SHA256"
```

构建工具：仓库附带已核对身份的 xmake 3.0.5 SDK、Apache 2.0 许可证及逐文件 SHA 清单，仅作为构建工具使用。它不是预编译的 InfiniCore/InfiniLM。准备脚本检查 SDK 身份后复制到独立工作目录。框架及 Python 扩展需要重新编译；NineToothed 生成源码中的 gfx936 设备字节码仍复用，不能称作重新生成了全部设备代码。

依赖脚本只安装到新建私有 venv，保留系统海光 Torch 和 NumPy 的版本与位置；从公开 PyPI 获取推理依赖，单独校验 torchvision CPU wheel。CPU 评分依赖另列在 `requirements-comparison.txt`，无需把科学计算包全部打进 GPU 运行产物。

在平台不能直接获取 GitHub 时，`make_source_bundle.py` 可以从已提交且干净的两套 checkout 制作传输包，记录每项源码、子模块 commit 和 SHA。`build_and_pack.py` 在共享 CPU 的独立临时目录执行准备、依赖、框架、CCL 四阶段，保存日志和新构建产物身份。`restore_rebuilt.py` 只接受该源码包对应的新产物，不接受历史 runtime 代替；迁移后再由运行入口验证实际加载。此方式解决传输与临时目录归档限制，不构成跳过编译或设备验证。

以上为可审查的复现候选。仍需记录本轮实际编译、视频、原 compare 输出后，才能把结果填入阶段报告的完整命令栏。


### 共享 CPU 首次执行结果（2026-11-10）

公开源码包身份和 prepare 阶段已实际通过；依赖安装尚未通过，框架编译尚未进入。首次 pip 的可选系统信息探测读取 `/etc/debian_version` 被权限拒绝；只省略 HTTP User-Agent 的系统分布信息后，已能获取多个包元数据，但平台代理出现 502/握手超时，Accelerate 索引未成功返回。不能把网络错误认定为版本不存在。无新增 GPU 作业。

新增离线候选：`install_dependencies.py --wheelhouse /path/to/wheels` 使用 `--no-index`，保留海光 Torch/NumPy 约束。`build_and_pack.py` 检测同目录的 `reviewer-offline-wheels.tar.gz` 与 `reviewer-offline-wheels.sha256`，验证归档及全部 wheel SHA 后使用；为控制项目归档大小，将输入 wheel 归档移到本轮临时构建根目录（原平台项目版本不变）。离线候选已主机检查，尚待目标安装/构建。未增加或替换模型计算算子。


### 离线目标执行补充

离线安装还遇到平台拒绝读取 `/etc/mime.types`；仅让安装子进程使用 Python 内置 MIME 表后，实际得到 `DEPENDENCY_INSTALL_PASS`，且系统 Torch/NumPy 身份检查通过。修正不读取受保护的文件、不修改系统权限。QK host 包装与 Core 配置也已通过，Core 主体在额外的 600 秒续跑预算到期时尚未完成，不能认定编译器失败或清洁构建已通过。该临时目录到期后已确认不可恢复。

历史成功记录中 Core 主体约 830–886 秒，另有 Python 扩展约 53–76 秒、LM 约 150–228 秒，因此下一轮直接使用完整构建 driver 的 1500 秒预算，不使用这次的 600 秒辅助续跑脚本。原失败及补充诊断分别保留，未生成新视频或新增 GPU 作业。


### OpenMP搜索路径补充（2026-11-10）

用户新执行409183d源码版本后，prepare与离线dependencies均退出0；框架阶段在354.67秒明确退出1，首个错误为Clang编译common_cpu.cc时 `omp.h` 找不到。头文件已在带许可证的tooling/openmp/include中，历史成功环境额外导出了CPLUS_INCLUDE_PATH和C_INCLUDE_PATH；新准备脚本遗漏这两项。现已补齐；同一目标Clang对包含omp.h的微型源文件做语法检查退出0、无stderr。该检查只证明头文件路径修正有效，完整框架/CCL/视频/比较尚未通过。
