# 性能分析

性能分析必须建立在同条件的正确性结果之上。模型质量、服务性能、设备区间和算子收益分别记录，不能用其中一项替代另一项。

## 服务性能

当前仓库通过实际引擎 CLI 执行服务 benchmark。采样前在目标环境运行：

```bash
python -B -m vllm.entrypoints.cli.main bench serve --help
```

固定模型、权重、模板、输入长度、输出长度、请求数、warmup、并发、采样参数和设备。保留实际 argv、原始 JSON、详细结果及失败请求。部署矩阵的固定合同是每组 `requests=32`、`warmups=2`、6 组 workload；改变合同必须创建新的实验并说明原因。

报告同时给出 TTFT、TPOT、ITL、端到端时延、吞吐和失败数。设备算子区间不能直接写成请求时延收益；强制输出长度的性能数据也不能代替正常生成质量。

## 设备 profile

设备采集由模型 runner 调用 `src/fullnet/measurements.py` 的 `collect_profile` 完成，仓库没有独立的 `fullnet profile` 子命令。服务侧 profiler 必须覆盖目标服务进程；只包裹 HTTP 客户端通常不能得到远端服务的 device kernel。

采样前检查当前 `msprof` 或 `torch_npu` 的 `--help`，把预热、采集区间、开始/停止条件和输出目录写入任务记录。任务至少保留：

- `performance/profile-commands.json`：实际调用 argv、退出码和失败阶段；
- `performance/profile-artifacts.json`：原始文件和设备证据索引；
- `trace_view.json` 或 `kernel_details.csv`：原始设备 trace 或 kernel 表。

设备 profile 只有在索引发现正时长的设备 kernel 事件，或 kernel CSV 中存在带设备/任务类型的非空 kernel 行时才接受。CPU trace、空 CSV 和无法识别设备字段的文件保留在原路径，但状态记为 `not_accepted`。

profile daemon 或解析失败时，保留 task 副本和原始 trace，从采集目录离线恢复。先用现场版本的 `msprof --help` 确认可用的离线 `import` 与 `op_summary` 方法，再沿 `profile-commands.json` 记录的实际参数生成索引。离线恢复出的行数和设备字段需要写回 `profile-artifacts.json` 并重新检查；它不能替代修复后的远端重跑，也不能把 CPU trace 变成 device profile。

## 归因与对照

按服务时间线拆分设备计算、通信、Host 提交、空闲和端到端阶段。profile replay 只提供离线归因，和线上服务性能分开报告。

原生与替换保持同一模型、输入、输出、并行、设备和采样次数。先确认替换命中和精度，再比较算子设备时间、整网请求时间、TTFT/TPOT、吞吐、显存和失败数。换卡或改变并行布局后，原生基线与候选必须整组重测；历史结果不能充当当前卡的测量。
