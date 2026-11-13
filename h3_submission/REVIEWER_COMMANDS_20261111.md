# MiniMax-H3 公开源码复现说明

本文件保留原入口文件名。2026-11-13 的 Fabric 使用说明已统一到
[Fabric 分阶段复现指令](FABRIC_REPRODUCTION.md)，以该文件为准。

修订内容：实际环境中可用的 Python 依赖与厂商 Flash 优先复用；缺少的包由评测者自己的离线 wheelhouse 补充；QK 静态库从公开源码随 xmake 自动构建；包元数据查询不再触发 C++ 编译。

评审需固定报告列出的三个公开仓库、分支和完整 commit；Core 使用 `minimax-h3`，LM 与 NineToothed 使用 `minimax_h3`。固定依赖仓库身份见 [dependencies.json](dependencies.json)。

入口仍为 `tools/h3_submission/reviewer_reproduce.sh`，分为 `prepare`、`dependencies`、`comparison-dependencies`、`flash`、`framework`、`ccl`、`verify`、`communication`、`native`、`reference`、`compare`，不自动申请资源或连续运行全部阶段。

没有新增 DCU 全链路运行记录；修订后的构建与离线流程仍需目标环境验证。不要将历史运行包或历史分数改称此提交的冷启动验收结果。生成 worker、VAE/cast、设备分组和原 compare 脚本未改变。
