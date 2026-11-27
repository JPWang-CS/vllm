# 项目启动与恢复

1. 先读根 `AGENTS.md`，再按任务读取所需 skill；部署/精度任务读
   `.agents/skills/fullnet-validate/SKILL.md`，故障定位读
   `.agents/skills/fullnet-debug/SKILL.md`，其他 skill 只按实际需要读取。
2. 若 `TASK_CONTEXT` 提供已有 `business_run`，先读取该 run 的
   `BUSINESS_CHECKPOINT.json`，再读取存在的 `repair/OPERATOR_UPDATE.json`。
3. 顶层旧 `CHECKPOINT.json` 和历史报告只作背景，不能覆盖当前 business run，
   也不能代替当前现场状态。
4. 查询已登记任务状态时，使用现有入口，不遍历目录猜测 raw response/schema：
   `python <absolute-root>/scripts/ops/recover_framework.py --project <absolute-root> --task <task-id> --action fleet-job-status --host <host> --job <job> --wait-seconds 15`
5. 完整原件按任务目录保存；汇报只输出 `job_outcome` 的关键字段，并保留可追溯的命令和退出结果。
6. `pending` 不等于 `queued`，`submit` 不等于 `passed`；状态未确认时不得推进为通过。
7. 明确下一步后推进一个实际实验，避免对同一资料重复只读；未知现场先查证。
8. 独立 case 可以并行，但必须保持项目目录边界、现有用户权限和其他任务隔离，不扩大授权。
9. 计算任务与 GLM 运行可以并行，但启动握手必须串行：一个 canonical 启动后，先核对自己的 `SUBMITTED`、代理真实存活或端口绑定归属，以及工具 `Read`/`Bash`（至少确认代理绑定归属），再启动下一个。driver 返回不等于代理就绪。
10. 若出现代理退出或 `remote port forwarding failed for listen port`，先查远端真实进程是否存活、是否已产生 business run，再决定恢复；按 UID、startticks、完整命令和任务归属精确 TERM/清理，禁止盲目重提，并保持活跃 profile 与其他任务不受影响。
