# 历史构建依赖原始记录

三份原始记录于2026-11-14从个人AFS的 `h3-rebuild-20261110-resume/persistent-round6/build` 下载，按原字节保留，SHA见 provenance.json。

这是历史依赖准备证据，**不是本次最终公开提交的干净编译结果**。不要根据这些文件宣称新构建已经通过。它们不包含二进制库、模型或wheel文件。

- dependency-install.json：原始 pip 安装报告，保留包元数据、来源及SHA；其中 `file:///dev/shm/...` 是当时的离线wheel路径，不是评审可访问的下载地址。
- requirements-private-resolved.txt：原始私有环境 freeze；不是完整基础镜像或完整传递依赖锁。torchvision 原始条目包含临时文件路径，**不要直接用此文件执行安装**。
- requirements-private-pinned.txt：仅将该 torchvision 文件路径改为已记录版本的便携版本摘录；仍不是完整系统锁文件。
- reviewer-base-constraints.txt：当时的海光Torch与NumPy约束（2.9.0+das.opt1.dtk2604、1.25.0），不能据此假定新的CPU/DCU镜像身份相同。
- flash-identity.json：历史厂商Flash身份摘要，不是新的动态加载或DCU执行记录；后续仍须校验实际.so与强制Flash门禁。

正式复现请使用 ../FABRIC_REPRODUCTION.md 和 tools/h3_submission 下的准备/安装入口。安装入口保护镜像的Torch/NumPy，在实际工作venv检查依赖；缺包时使用评审自己准备的离线wheelhouse，不读取作者临时路径。本次全新验证的阶段报告、日志及产物SHA将另行记录，不能以本目录历史证据替代。
