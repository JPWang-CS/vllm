# 部署矩阵与验收

`scripts/ops/deployment_matrix.py` 负责生成、提交、刷新和审计部署矩阵；实际设备查询、容器任务和业务 tmux 由 `recover_framework.py` 的受控入口执行。矩阵计划只说明候选项，不能作为设备或模型通过证据。

## 单机提交

```bash
python -B scripts/ops/deployment_matrix.py --project . --task <task> --action plan
python -B scripts/ops/deployment_matrix.py --project . --task <task> --action status
python -B scripts/ops/deployment_matrix.py --project . --task <task> --action submit --case <case>
python -B scripts/ops/deployment_matrix.py --project . --task <task> --action refresh
```

A5 单机 case 使用受控 native runner。提交前只读核对权重、48 条冻结数据、参考配置、CANN、模型运行时、算子目录和实际加载路径；随后重新检查设备健康、空闲状态及物理到逻辑映射，再提交独立 tmux 任务。资源不足、配置不完整或上一任务未清理时停止提交。

设备选择是动态的：可用卡必须同时满足健康、无本任务外的运行进程且不在保护列表。换卡不沿用旧卡结果；原生基线、固定评测、6 组性能和设备 profile 需要整组重测。

## 多卡、多机和 P/D

A5 单机 TP2、TP4、TP2DP2、TP1DP4 展开 Eager/Graph 与 MTP0/3；每个 case 使用独立实验。A3、multi-host TP/DP、same-host split PD、跨机 PD 和 mixed deployment 分开建 case。`fleet-inventory` 只收集登记主机的只读事实，不能解锁没有真实 adapter、权重/runtime、host role、设备映射和可执行配置的 case。

```bash
python -B scripts/ops/deployment_matrix.py --project . --task <task> --action fleet-inventory --case <case> --hosts <registered-host-1> <registered-host-2>
python -B scripts/ops/deployment_matrix.py --project . --task <task> --action fleet-status --case <case>
```

跨机网络、Ray 或 HTTP fixture 通过，只能证明相应基础设施或编排路径；没有权重、运行时、源码和真实设备执行，不写成 multi-host TP 或 P/D 通过。同机 PD 与专项 mixed 需要各自的角色启动、设备分组、服务交互和完整验收。

## 通过条件

单个 case 只有全部满足才记为 `passed`：

- 业务 job 真实退出码为 0，experiment execution 和 evaluation 状态通过；
- 冻结配置与 case 的 platform、TP/DP/PP、graph、MTP 一致，48 条样本均尝试、成功 48、失败/跳过/超时为 0，评分协议保持原样；
- 六组固定服务基准每组 `requests=32`、`warmups=2`，且每组请求完成；
- profile 索引存在正时长的 NPU/device kernel 事件或有效 kernel CSV，CPU trace 不通过；
- 任务结束后选中设备、服务端口和本任务 worker 均释放。

得分阈值未声明时记录得分和 `quality_status=not_specified`，不把 HTTP 请求全部返回写成质量达标。指标缺失、设备占用、启动/评测失败、profile daemon 失败或清理未确认分别保留为 `incomplete`、`blocked_resource`、`failed` 等状态。

通信错误需要区分通信组初始化成功和第一个 TP AllReduce/AllGather 失败。首个 collective 失败时保留 rank、设备映射和原始日志，根因未确认前不归咎确定的 vendor bug。所有调试和恢复都禁止 NPU 带外 reset、驱动变更和公共环境修改。

`fullnet run` 返回成功只表示本次框架任务完成。精度、服务性能、设备 profile 和资源清理分别验收；HTTP fixture 的编排结果不能替代真实模型或设备结果。结果查看使用 `fullnet status`、`fullnet report` 及实验目录中的原始状态和报告。
