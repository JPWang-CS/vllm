# 长任务与 AI 执行

模型下载、加载和评测由项目的独立 tmux 会话运行。AI 的一次 Bash 调用只负责提交；退出码、原始日志和实验结果由任务自身保存。这样，AI 工具超时或 SSH 断开后，业务任务仍能继续。

在已确认的项目目录中准备一份任务配置，`argv` 使用参数列表。例如提交已有实验：

```json
{
  "argv": [
    "/absolute/project/envs/fullnet/bin/fullnet",
    "run", "--model", "pangu-7b", "--run", "a5-native-001"
  ]
}
```

把配置保存为项目内的 `.local.json`；路径填写本机的真实值。先完成 `fullnet check` 和 `plan`，再进入已加载 CANN 的个人容器，在项目目录执行：

```bash
mkdir -p runs/a5-native-001
python -B scripts/ops/submit_job.py \
  --project "$PWD" --task a5-native-001 \
  --config "$PWD/models/pangu-7b/configs/job.local.json" \
  --tmux "$(command -v tmux)"
```

提交回执给出 socket 和状态路径。任务使用 `.runtime/jobs.sock`，日志集中在 `runs/a5-native-001/`：

| 文件 | 内容 |
|---|---|
| `job-config.json` | 本次固定的命令参数 |
| `job-state.json` | 当前状态、进程与起止时间 |
| `job.log` | 业务命令的标准输出与错误 |
| `job-exit.txt` | 业务命令实际退出码 |
| `runner.log` | 监督程序自身的错误 |

`submitted=true` 表示提交成功。任务完成后核对 `job-exit.txt`，再读取模型实验中的 `results.json` 和清理记录。AI 会话退出、任务命令退出和模型精度是三项独立结果。

恢复前先检查原任务的状态与 tmux 会话。下载沿用同一模型缓存；模型实验保留已经固定的配置和输入，新尝试使用新的实验编号。任务配置与日志保持在项目内，认证通过安全环境传入。

停止实验时，根据本次实验的 `process.json` 核对服务身份，向本次监督进程发送正常终止信号，并继续检查服务及 Worker 是否退出、设备和端口是否释放。宿主上的 `docker exec` 进程退出后，仍需检查容器内的业务进程。
