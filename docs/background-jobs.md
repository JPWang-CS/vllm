# 长任务与 AI 执行

模型下载、加载、评测和 profile 使用项目独立 tmux。AI 的一次 Bash 调用只负责提交；业务任务保存自己的退出码、原始日志和实验结果，因此工具超时或连接断开不会被误判为业务结束。

## 提交任务

先完成 `fullnet check` 和 `fullnet plan`，再在已加载目标运行时的个人容器中准备项目内任务配置。`argv` 使用参数列表，不把认证信息写入配置：

```json
{
  "argv": [
    "/absolute/project/envs/fullnet/bin/fullnet",
    "run", "--model", "pangu-7b", "--run", "a5-native-001"
  ]
}
```

使用仓库已有的提交入口：

```bash
python -B scripts/ops/submit_job.py \
  --project "$PWD" --task a5-native-001 \
  --config "$PWD/models/pangu-7b/configs/job.local.json" \
  --tmux "$(command -v tmux)"
```

提交器使用项目内 `.runtime/jobs.sock`，拒绝重复 task，并在任务目录保存：

| 文件 | 内容 |
|---|---|
| `job-config.json` | 本次冻结的命令参数 |
| `job-state.json` | 业务状态、进程和起止时间 |
| `job.log` | 业务标准输出与错误 |
| `job-exit.txt` | 业务命令的真实退出码 |
| `runner.log` | 监督器自身的错误 |

`submitted=true` 只表示任务已经提交。完成后读取 `job-exit.txt`、`job-state.json`、模型实验的 `results.json`、`validation.json` 和清理记录；AI 会话退出、控制器退出、业务退出和模型验收分别判断。

## 控制器、broker 与业务状态

控制 GLM 或提交工具负责发起受控请求，业务任务仍由独立 tmux 运行。控制器的终态必须同时看到真实退出码、子进程已退出、临时通道已清理和控制凭据未被持久化；中间状态文件、工具返回或一次心跳都不能作为完成信号。

broker heartbeat 只说明 broker 仍可报告状态，不说明模型服务、评测或 profile 正在运行。需要以 `job-state.json`、`job-exit.txt`、业务日志、实验状态、端口和设备查询共同判断。

新实验的 `acceptance.json` 汇总必做阶段及其证据。`execution_completed`、`quality_status` 和 `business_acceptance` 分别表示协议完成、评分结论和完整验收，具体含义见[实验阶段与验收](acceptance-contract.md)。旧实验继续展示原记录。

## 恢复与停止

恢复前先检查原任务状态、对应 tmux 会话、业务进程、端口和设备。不要重复提交仍在运行或状态未确认的 task；新尝试使用新的实验编号，保留已经冻结的配置和输入。

停止实验时，根据本次实验的 `process.json` 核对服务身份，先向本任务监督进程发送正常终止信号，再确认服务、worker、profile、端口和设备占用已经释放。宿主 `docker exec` 退出不等于容器内业务进程退出，也不等于设备清理完成。

只清理本任务的 tmux、锁、代理和临时运行资源，保留源码、配置、模型、日志和结果。遇到公共服务、他人进程、权限边界或 NPU reset 需求时停止并交回环境管理。
