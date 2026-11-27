# Fullnet Lab

仓库用于整网运行、精度验证、性能分析及算子替换实验。Python 包为 `fullnet`，与推理引擎 `vllm` 分开安装。

先读 README 和目标 `models/<model>/README.md`，再读取所需项目 skill：

- `.agents/skills/fullnet-validate/SKILL.md`：环境核对、启动与精度。
- `.agents/skills/fullnet-replace/SKILL.md`：外部算子接入和原生对照。
- `.agents/skills/fullnet-profile/SKILL.md`：整网与设备性能采集、归因。
- `.agents/skills/fullnet-debug/SKILL.md`：根据原始失败定位环境、加载、执行或数值问题。
- `.agents/skills/fullnet-optimize/SKILL.md`：把整网热点转为算子或系统的可验证优化。

源码、临时文件、缓存和日志写在当前项目内；权重和登记环境按只读资产使用。A3、A5 分开配置和验收。操作前读取环境管理的服务器规范，通过获准连接进入个人容器。长任务使用项目独立 tmux，禁止改公共 CANN、驱动、编译器及其他人的任务；禁止 NPU 带外热复位。

模型、配置、算子变更、基线和实验按 [模型与实验目录规范](docs/repository-layout.md) 组织。所有实验归属 `models/<model>/experiments/<run-id>/`；新增模型先建立 `model.yaml` 和 `README.md`，新增部署复用原模型目录和现有 CLI。原生 baseline 使用一个实验，替换 candidate 使用另一个实验；每个实验内按合同保存精度、性能和 profile，输入、配置或实现合同改变时才创建新 run。

用户指定本任务主会话完成本地架构，上机执行交给具备真实工具的 GLM。文本 worker 的答复不能作为服务器执行证据。提交脚本先审核，回收原始工具事件、退出码、请求响应及清理记录。

`models/<model>/experiments/<id>/config.json` 和 `dataset.jsonl` 在启动前冻结。变更后的输入、模板、生成参数或配置创建新实验。检查旧进程和状态后恢复，不能覆盖旧结果。服务请求通过、任务精度、数值一致性、性能分别记录。

不自动 commit/push，不记录认证信息，不做例行文件哈希。实现后运行项目内测试及离线流程，再进行真实设备验收。

新任务启动前同步已发布 main 的框架源码，并核对实际加载路径。远端源码副本使用项目内的已发布快照与 `.runtime/framework-source.json`，按实际内容核对；不得仅凭分支名称或文件存在判断更新完成。旧 broker 的内存源码与磁盘不一致时拒绝新 run；已有运行任务不热更新、不重启。源码差异先保存在本任务内，不修改权重、环境、私有配置或实验记录。
