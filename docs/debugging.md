# 调试方法

Fullnet Lab 的调试以首个可复现错误为边界。先保留原始配置、实际加载路径、阶段日志和退出码，再决定修复范围；修复后使用新实验 ID，旧实验只读保留。

## 先确认运行对象

读取实验的 `state.json`、`validation.json`、冻结 `config.json`、服务日志和任务状态。用 `fullnet status --model <model> --run <id>` 查看汇总，用 `fullnet report --model <model> --run <id>` 生成报告；两者都不能代替原始日志。

首先确认四件事：

1. 原始配置选择了哪个 Python、CANN、平台插件、权重和环境变量。
2. 实际 import 文件和动态库来自哪里。项目已有 `scripts/ops/container_inventory.py`、`scripts/ops/probe_model_stack.py` 记录版本、工具和加载来源。
3. 设备是否健康且空闲，物理卡如何映射到运行时逻辑卡，本任务是否占用端口或设备。
4. 失败发生在环境检查、模型加载、通信初始化、首个请求、固定评测、服务 benchmark 还是设备 profile。

设备动态选卡时记录完整映射；“有一张空闲卡”不等于当前并行布局可运行。换卡后，原生基线、冻结的评测与性能负载、设备 profile 必须整组重测。

## 按阶段缩小范围

先执行一次真实短生成，确认服务已经完成加载并能返回业务响应，再进入固定评测。固定评测保持原始样本、模板、输入/输出预算、种子、评分函数和分母；性能阶段保持各组请求数、warmup、输入/输出长度和并发。当前盘古代表验证使用 48 条评测与 6 组性能，其他模型以登记的冻结合同为准。未完成的新续跑记录为 `incomplete` 或 `pending`，不写 `pass`。

通信失败需要区分“通信组初始化成功”和“第一个 TP AllReduce/AllGather 失败”。保存 rank、物理/逻辑设备映射、初始化日志和第一个 collective 的原始错误。根因未确认前，不把现象归咎为确定的 vendor bug，也不通过更换驱动或 reset 设备绕过问题。

## Profile 失败的恢复

设备 profile 由模型 runner 调用 `src/fullnet/measurements.py` 完成，不新增虚构的 `fullnet profile` 命令。任务中保留 `performance/profile-commands.json`、`performance/profile-artifacts.json`、原始 trace 和 kernel CSV。当前索引识别 `trace_view.json`、设备 kernel CSV 及 `op_summary` CSV，只有正时长设备 kernel 事件或带设备/任务类型的非空 kernel 行才构成设备证据；CPU trace 不通过。

如果 profile daemon 或解析阶段失败，保留完整 task 副本，从原始采集目录做离线恢复。使用现场版本 `msprof --help` 确认可用的离线 `import` 和 `op_summary` 方法，沿用任务中记录的实际 argv，生成或补全 profile 索引，再检查设备字段、正时长和行数。离线索引恢复只能修复证据处理链，不自动把原失败任务改成通过；修复后的远端 runner 必须重新执行并完成完整验收。

## 修复与收尾

项目私有目录内可以自主完成最小修复。修改后创建新实验，保持原输入和合同不变，逐阶段复测并保留失败分支。后台任务通过 `scripts/ops/submit_job.py` 进入项目独立 tmux，由 `scripts/ops/job_runner.py` 保存 `job-state.json`、`job-exit.txt` 和业务日志。控制工具退出、broker 心跳、业务进程退出、端口释放和设备空闲分别核对；心跳不是业务运行证据。

遇到公共环境、他人任务、权限边界、驱动变更或 NPU 带外 reset 需求时停止并交回环境管理。只清理本任务的服务、worker、proxy、端口和 tmux 资源。
