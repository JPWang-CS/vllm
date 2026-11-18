# Pangu 92B 部署矩阵验收

`scripts/ops/deployment_matrix.py` 负责生成、提交和刷新部署矩阵；实际 NPU 查询、容器任务及业务 tmux 仍由 `recover_framework.py` 的受控入口执行。矩阵计划只说明待验项目，不能作为设备或模型通过证据。

## 创建与单机提交

```bash
python -B scripts/ops/deployment_matrix.py --project . --task deployment-matrix-20261117-01 --action plan
python -B scripts/ops/deployment_matrix.py --project . --task deployment-matrix-20261117-01 --action status
python -B scripts/ops/deployment_matrix.py --project . --task deployment-matrix-20261117-01 --action submit --case a5-single-tp2dp1-eager-mtp0
python -B scripts/ops/deployment_matrix.py --project . --task deployment-matrix-20261117-01 --action refresh
```

A5 单机 native case 使用现有 `run_pangu92b.py`，提交前通过个人容器只读核对权重、48 条冻结数据、参考配置、CANN 路径、模型运行时与算子目录；随后由受控 recovery 入口检查设备健康/空闲并提交单独 tmux 任务。资源不足或资产不完整会写成 `blocked_resource` / `blocked_dependency`，不会启动服务。

A5 单机 TP2、TP4、TP2DP2、TP1DP4 分别展开 Eager/Graph 和 MTP0/3，case 都显式记录 `model=pangu-92b`。Pangu 7B 的 A3/A5 单卡 case 单独记录 `model=pangu-7b`；现有 `validate_pangu7b.py` 可完成质量验证，但目前没有把它接入 32+2 性能和真实设备 profile 的完整矩阵 runner，因此尚为 `blocked`。

A3、multi-host TP/DP、PD mixed、same-host split PD 与跨机 PD 分开建 case。`pd_mixed_single_instance` 表示同一个服务实例同时处理 Prefill 和 Decode；`same_host_pd_split` 表示同机独立 Prefill 与 Decode 服务进程；`pd_disaggregated` 表示两台主机分别承载 P/D。当前没有 A3 Pangu 92B launcher、host-rank/rendezvous 参数或 P/D 前后端服务适配器，这些项保持 `blocked`。不得用 A5、HTTP fixture 或“计划已生成”替代对应平台的实际运行。

## Fleet inventory 与资产

GLM 可为 multi-host/P-D case 创建只读 inventory 请求：

```bash
python -B scripts/ops/deployment_matrix.py --project . --task deployment-matrix-20261117-01 --action fleet-inventory --case a5-multi-host-tp-eager-mtp0 --hosts <registered-ip-1> <registered-ip-2>
python -B scripts/ops/deployment_matrix.py --project . --task deployment-matrix-20261117-01 --action fleet-status --case a5-multi-host-tp-eager-mtp0
```

需要复制已有权重时，只能用 `recover_framework.py --action fleet-submit --fleet-action copy_asset --asset pangu-7b|pangu-92b` 生成请求；broker 在受信工作站核验注册主机、源文件和真实目标项目路径，按文件尺寸复用，不下载、不写共享环境。GLM 不能读取 SSH 凭据。所有请求/回执留在 `runs/<task>/fleet/{requests,responses}/`。

fleet inventory 结果本身也不能解锁 multi-host/P-D 运行。只有真实 adapter、权重/runtime、host role/地址、设备映射和可执行配置均准备好后，GLM 才能在任务 repair 目录编写受限 runner，通过 fleet `run` 请求提交；端到端结果仍按完整质量、性能和 profile 合同验收。Broker 的 run 门禁查询 `npu-smi info -t proc-mem -i <id>`，现场该接口已返回 `No process in device.`，与 broker 当前检查一致；这与 `npu-smi info` 大表中的 `No running processes found in NPU <id>` 属于不同查询接口，不应混作兼容故障。

已存在的权重可按目标 host 请求尺寸匹配复制，例如 `recover_framework.py --action fleet-submit --request-id copy-92b-01 --fleet-action copy_asset --host <registered-ip> --job <safe-job-id> --asset pangu-92b`。broker 只从控制项目的 `assets/weights/<asset>` 复制登记模型分片到目标私有项目；缺失资产不会伪造占位文件，目标已存在且文件大小不同则停止等待人工核对。

## 通过条件

单个 case 只有全部满足才记为 `passed`：

- 业务 job 真实退出码为 0，experiment execution/evaluation 状态通过。
- 冻结配置与 case 的 platform、TP/DP/PP、graph、MTP 一致；原始 48 条数据均尝试、成功 48、失败/跳过/超时为 0，评分协议通过。
- 六组固定服务基准每组 `requests=32`、`warmups=2` 且 32 个请求完成。
- profile 状态通过，索引指出存在有实质内容的 NPU kernel 事件/CSV；CPU trace 不通过。
- 任务结束后选中设备空闲证据通过。

指标缺失、设备占用、启动/评测失败或清理未确认分别保留为 `incomplete`、`blocked_resource`、`failed` 等状态。case 输出、每次 attempt 与首次错误不覆盖；重试必须使用新实验 ID。
