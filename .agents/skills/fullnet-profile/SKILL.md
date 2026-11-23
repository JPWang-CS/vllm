---
name: fullnet-profile
description: 对已通过正确性检查的模型实验采集整网与设备性能，比较算子替换收益并形成优化假设。
---

# 整网性能与算子归因

先读取冻结配置、精度结果、服务参数和基线引用。当前仓库没有独立的 `fullnet profile` 子命令；设备采集由模型 runner 调用 `src/fullnet/measurements.py` 的 profile 方法完成，结果写入实验目录。

1. 用实际环境的 `python -B -m vllm.entrypoints.cli.main bench serve --help` 核对 benchmark 参数。保存实际 argv、原始 JSON、请求数、warmup、输入/输出长度、并发、TTFT、TPOT、ITL、吞吐和失败数。每组性能按冻结的负载合同执行；当前盘古代表验证包含六组。
2. 采样前读取服务侧 profiler 接口和当前 `msprof`/`torch_npu` 的 `--help`。HTTP 客户端 trace 不能替代服务进程设备采集；预热、开始/停止条件和采集目录写入任务配置。
3. 任务目录至少保留 `performance/profile-commands.json`、`performance/profile-artifacts.json`、原始 trace 或 kernel CSV。仓库的设备证据索引识别 `trace_view.json`、设备 kernel CSV 及 `op_summary` CSV，并要求正时长的设备 kernel 事件或带设备/任务类型的非空 kernel 行；CPU trace、空文件和无法确认设备事件的文件不通过。
4. profile daemon 或解析阶段失败时保留完整任务副本，不删除原始 trace。使用当前环境实际支持的 `msprof` 离线 `import` 和 `op_summary` 方法恢复索引；具体 argv 以任务记录和现场 `--help` 为准，不把跨版本命令写死。离线恢复出的行数仍需回填 `profile-artifacts.json` 并重新检查设备字段。
5. 把设备计算、通信、Host 提交、空闲和端到端服务时间分开报告。profile replay 只说明离线归因，不等于线上收益；设备区间也不等于 TTFT、TPOT 或完整请求时延。
6. 原生与替换固定模型、权重、输入、输出、并行、设备和采样条件。先保存原始证据，再提出一个可验证改动；报告收益、退化、适用形状、样本数和未测范围。

采样和模型服务由项目私有 tmux 管理。结束后只检查并清理本任务进程、端口和设备；换卡或改变并行布局后，原生基线和候选结果整组重测。
