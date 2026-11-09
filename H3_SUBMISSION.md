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
source h3-build/setup.sh
python InfiniLM/tools/h3_submission/build_framework.py

# 补启通信后才能做真实 TP；不能拿 --ccl=n 的基础构建当作 TP 就绪。
python InfiniLM/tools/h3_submission/build_ccl.py --plan
python InfiniLM/tools/h3_submission/build_ccl.py
```

构建会在新目录编译主机框架、Python 扩展、QK 归一化等 AOT 包装。已有 gfx936 设备代码以生成 C/C++ 中的嵌入数组保留，**本步骤复用它们，不重新生成全部设备内核，也不要求仅为本构建安装 Triton**。`build/ninetoothed*` 在此版本中包含生成源码，不能把这些目录当成普通输出全部删掉。

证据边界：`build_framework.py` 保留实际运行包中的历史构建脚本；该基础构建当时 `--ccl=n`，随后实际测试用独立 CCL-on 库覆盖。`prepare_build.py` 是为公开仓库新整理的目录准备脚本，`build_ccl.py` 是依据当前 xmake 目标整理的补建方案，**本轮未在 DCU 执行这两个新脚本，未取回历史 CCL-on 精确命令，也未声称这套新整理流程已干净重建成功**。源文件及原视频运行证据与这个尚待目标环境验证的交付流程分开记录。

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
