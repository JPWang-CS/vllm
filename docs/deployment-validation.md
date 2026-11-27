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

设备选择是动态的：配置里的 `devices` 表示模型进程使用的 runtime/logical selector，不表示 `npu-smi` 的物理 board。可用卡必须先取得当前 host 的物理到 ACL user/runtime 映射，再同时满足健康、无本任务外的运行进程且不在保护列表；Slot ID、Chip Phy-ID、表格行号和 runtime ID 不能互相替代。这个 fresh gate 必须在启动前完成，不能用历史 mapping、旧 `device-post.txt` 或旧任务状态代替。换卡、换 host 或换平台不沿用旧卡结果；原生基线、固定评测、6 组性能和设备 profile 需要整组重测。

A2、A3、A5 的基础功能都使用独立 case：真实服务启动、短请求、冻结评测和 cleanup 必须分别成立。A2 不因 CLI 支持 `platform=a2` 就继承 A3/A5 的设备、映射、分数或性能结果；每个平台都要在自己的 runtime、权重、CANN 和 fresh gate 上重新验证。

## 公开 CLI 的测量与 P/D 入口

公开入口可选择以下阶段；split 的 KV 检查在固定评测前执行，mixed、profile 和 performance 在固定评测后执行：

```bash
fullnet run --model <model> --run <run> --profile --performance
fullnet run --model <model> --run <run> --pd-probe mixed
fullnet run --model <model> --run <run> --pd-probe split
```

使用 `--profile` 或 `--performance` 前，冻结配置必须包含合法的 `measurements` mapping。盘古代表 contract 至少应包含：

```yaml
measurements:
  requests: 32
  warmups: 2
  timeout_per_group: 21600
  deferred_stages: []
  workloads:
    - {input: 128, output: 128, concurrency: 1}
    - {input: 4096, output: 128, concurrency: 1}
    - {input: 15999, output: 128, concurrency: 1}
    - {input: 128, output: 1024, concurrency: 1}
    - {input: 4096, output: 1024, concurrency: 4}
    - {input: 128, output: 4096, concurrency: 4}
```

这 6 组 workload 中最大的 `input + output` 是 `15999 + 128 = 16127`，因此 service argv 的 `--max-model-len` 必须至少为 16127；当前 A2/A3/A5 Pangu 7B 示例使用 16384。任何新 workload 或上下文上限都必须逐组满足 `input + output <= max_model_len`。

`--profile` 推荐通过配置的 `env.VLLM_TORCH_PROFILER_DIR` 启用服务端 profiler。当前 CLI 也会把 service argv 中带非空值的 `--profiler-config <value>` / `--profiler-config=<value>` 作为前置存在性信号，但不会解析文件、JSON 字符串或目标 vLLM service 的参数语义；这两种 argv 拼写不能被当作真实 profiler 已启用的证据。使用该方式前必须用目标 service 的实际 `--help` 或源码确认支持的格式，并按目标接口传值。配置缺失、shape 非正整数或对应阶段仍在 `deferred_stages` 中时，CLI 会在启动模型前拒绝；修改后必须重新 `plan`，不能在已经冻结的 run 上静默补字段。`run_experiment` 会在服务启动前准备 profiler tools，callback 只负责评测通过后的 collector 调用。

`--pd-probe mixed` 只适用于同一个服务实例具备混合 P/D 证据的部署。`--pd-probe split` 只适用于已经启动 producer/consumer、KV-transfer connector、split front，并提供 `/metrics_snapshot` 和两侧 connector 日志的部署；普通单服务 argv 不是 split case。P/D 角色、拓扑、设备分组或服务入口变化时，另建 case 和 run。

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

得分阈值未声明时记录得分和 `quality_status=not_specified`，不把 HTTP 请求全部返回写成质量达标。基础功能/协议结果、质量结果、性能结果和 cleanup 是四个独立边界：48 条请求成功只能说明当前评测协议完成；只有存在明确阈值且得分满足阈值时，质量才是 `passed`。指标缺失、设备占用、启动/评测失败、profile daemon 失败或清理未确认分别保留为 `incomplete`、`blocked_resource`、`failed` 等状态。

通信错误需要区分通信组初始化成功和第一个 TP AllReduce/AllGather 失败。首个 collective 失败时保留 rank、设备映射和原始日志，根因未确认前不归咎确定的 vendor bug。所有调试和恢复都禁止 NPU 带外 reset、驱动变更和公共环境修改。

`fullnet run` 返回退出码 0 只表示框架任务和评测协议完成，不能单独解释为质量通过或完整部署通过。`quality_status=not_specified` 必须原样保留；外部 CI、交付脚本和人工验收还要检查 `quality_status`、profile、performance、cleanup、设备释放和本任务退出状态。精度、服务性能、设备 profile 和资源清理分别验收；HTTP fixture 的编排结果不能替代真实模型或设备结果。结果查看使用 `fullnet status`、`fullnet report` 及实验目录中的原始状态和报告。
