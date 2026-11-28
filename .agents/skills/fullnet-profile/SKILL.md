---
name: fullnet-profile
description: 对已通过正确性检查的模型实验采集整网与设备性能，比较算子替换收益并形成优化假设。
---

# 整网性能与算子归因

先读取冻结配置、精度结果、服务参数和基线引用。新实验由 `fullnet plan` 固定必做阶段，`fullnet run --model <model> --run <run>` 自动执行合同中的测量项；`--profile`、`--performance`、`--pd-probe mixed|split` 保留显式入口。P/D 的真实 KV 检查在 smoke 后、固定评测前完成。profile/performance 使用冻结的 `measurements`、负载和计数；功能优先的计划用 `--quality-only` 明确延后。阶段验收见 [实验阶段与验收](../../../docs/acceptance-contract.md)。

profile 优先配置 `VLLM_TORCH_PROFILER_DIR`；使用 `--profiler-config` 时按目标 vLLM 的 help 填写 JSON 或参数值。`run_experiment` 在真正 Popen 前，以实际 launch env 验证选定 Python 的 SQLite、官方 Python CLI 命令族和任务输出目录。缺配置时补齐后新建 plan，保留原实验记录。

恢复分母使用 `measurements.profile_expected_profs` 和 `profile_expected_devices`，未显式配置时沿用冻结 `devices` 合同；设备 metadata 与运行时编号不同时须在新 plan 中明确预期值。已有导出先执行 verify，不重复 import/export；CSV、完整流式 timeline 与 raw metadata 逐 rank 核对。HTML 共用时间轴须有共同原始时钟证据，先减共同零点再转换浮点；不能把各端归零后同步窗口解释为真实重叠。

1. 用实际环境的 `python -B -m vllm.entrypoints.cli.main bench serve --help` 核对 benchmark 参数。保存实际 argv、原始 JSON、请求数、warmup、输入/输出长度、并发、TTFT、TPOT、ITL、吞吐和失败数。每组性能按冻结的负载合同执行；当前盘古代表验证包含六组。
2. 采样前读取服务侧 profiler 接口和当前 `torch_npu` 版本及入口。先只读检查 SDK 的官方 Python 离线入口；`msprof` ELF 即使 `--help` 也可能初始化驱动。HTTP 客户端 trace 不能替代服务进程设备采集；预热、开始/停止条件和采集目录写入任务配置。
3. 任务目录至少保留 `performance/profile-commands.json`、`performance/profile-artifacts.json`、原始 trace 或 kernel CSV。仓库的设备证据索引识别 `trace_view.json`、设备 kernel CSV 及 `op_summary` CSV，并要求正时长的设备 kernel 事件或带设备/任务类型的非空 kernel 行；CPU trace、空文件和无法确认设备事件的文件不通过。读取恢复产物时完整流式遍历 CSV，逐 rank 绑定来源目录与设备 ID；两 rank 的设备绑定必须分别成立，不能用合并总数替代缺失 rank。巨型 timeline/trace 必须流式读取，允许 metadata 晚于事件但需在结束时完成绑定；根 array/object 的 JSON 分隔逗号、闭合和尾部必须严格校验，缺逗号、尾逗号、截断或尾随垃圾均失败。
4. profile daemon 失败、CPU-only trace 或索引为空时，先读 [离线诊断与恢复](references/offline-recovery.md)，运行 `scripts/profile_recovery.py inspect` 区分采集未确认、设备原始数据已有但导出失败、导出成功但索引遗漏。优先在新任务副本用官方 Python CLI 恢复，逐 rank 验 summary、设备 timeline、子进程错误与设备映射；保留原始 `PROF_*`，不重跑模型、不覆盖旧状态。若已有实质性 CSV 证据，生产索引可以记录并跳过重复解析超过阈值的大 JSON timeline，但必须保留 skip 原因；完整独立 timeline audit 仍单独执行、单独留证，不能用索引 skip 充当 timeline 通过。验收后的新索引交调用方更新 `profile-artifacts.json`，不是直接将旧失败改为通过。
5. 把设备计算、通信、Host 提交、空闲和端到端服务时间分开报告。profile replay 只说明离线归因，不等于线上收益；设备区间也不等于 TTFT、TPOT 或完整请求时延。
6. 原生与替换固定模型、权重、输入、输出、并行、设备和采样条件。先保存原始证据，再提出一个可验证改动；报告收益、退化、适用形状、样本数和未测范围。

采样和模型服务由项目私有 tmux 管理。结束后只检查并清理本任务进程、端口和设备；换卡或改变并行布局后，原生基线和候选结果整组重测。
