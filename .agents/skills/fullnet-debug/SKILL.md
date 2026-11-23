---
name: fullnet-debug
description: 读取模型实验的失败现场，定位启动、模型加载、通信、执行和精度问题，提出最小复现与修复。
---

# 整网调试

先保留现场，再解释首错。读取原始配置、冻结配置、`state.json`、`validation.json`、阶段日志、实际包文件路径和加载环境；`fullnet status` 与 `fullnet report` 只作为汇总，不能替代原始日志。

按以下边界推进：

1. 环境与加载：核对原始配置、Python/CANN、插件、权重和实际 import 路径。可使用已有的 `scripts/ops/container_inventory.py` 与 `scripts/ops/probe_model_stack.py`，只读保存版本、工具和来源。
2. 设备与映射：重新检查健康、空闲、物理卡到逻辑卡映射和本任务占用。动态选卡后记录完整映射；换卡必须重跑对应原生基线组，不把上一张卡的数据移作当前卡证据。
3. 执行链路：先核对冻结的评测、性能和设备 profile 合同，再做真实短生成；当前盘古代表验证使用 48 条评测与 6 组性能。每一阶段独立记录成功、失败、跳过和超时。
4. 通信：区分通信组初始化成功与第一个 TP AllReduce/AllGather 失败。首个 collective 失败时保留 rank、设备映射、日志和初始化结果；根因未确认时不归咎 vendor bug。
5. Profile：daemon 失败保留 task 副本，按 `fullnet-profile` 使用现场支持的离线 `msprof import` 与 `op_summary` 建索引。CPU trace 不能变成 device profile 通过。
6. 修复后创建新实验，保持输入、模板、阈值和分母不变。没有新一轮真实执行时，状态保持 `incomplete`/`pending`，不写 `pass`。

允许在项目私有目录内自主修复和复测；遇到公共环境、他人任务、权限、驱动或设备 reset 需求立即停止并保留证据。禁止 NPU 带外热复位。
