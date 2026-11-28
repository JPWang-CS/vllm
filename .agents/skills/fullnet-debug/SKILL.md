---
name: fullnet-debug
description: 读取模型实验的失败现场，定位启动、模型加载、通信、执行和精度问题，提出最小复现与修复。
---

# 整网调试

先保留现场，再解释首错。读取原始配置、冻结配置、`state.json`、`validation.json`、阶段日志、实际包文件路径和加载环境；`fullnet status` 与 `fullnet report` 只作为汇总，不能替代原始日志。

## 入口路径与 cwd

GLM 或任务脚本的起始 `cwd` 可能是 controller/task 目录，不要把它当作项目根。执行命令前固定做三步：

1. 从 `TASK_CONTEXT` 和 `RUN_IDENTITY` 读取并记录实际 `project`、`business_run`、`controller_run`；用 Bash 先核对 `pwd`，再用这些绝对路径定位文件。源码和脚本使用 `Path(project)/scripts`，业务 spec、结果和输入使用 `Path(business_run)`，控制器回执使用 `Path(controller_run)`。
2. 命令、spec、配置和脚本都传显式绝对路径；不要凭 `./`、当前 cwd 或 `repair/...` 猜项目根，也不要把 controller/task 目录中的同名文件当作 business 文件。
3. 文件不存在时，先按绝对路径逐层检查父目录、真实路径和目录归属，再判断是路径错误还是尚未上传；不得从其他项目复制同名文件，也不得通过改权限掩盖路径或挂载问题。

## 环境与恢复的固定检查

- profiler 准备记录不等于可解析：模型 Popen 前必须用实际 launch env 完成选定 Python 的 SQLite 与官方 Python CLI 命令族探测。解析失败优先恢复已有 raw；import、summary、timeline、显式 rank/device verify 各自留收据，不能因父 exit0 或已有一端 CSV 通过而交付。

- broker 脚本的非代码依赖放在 `repair/staging-manifest.json` 或脚本同名 `.dependencies.json` 的 `files` 列表中（路径从任务根开始，如 `repair/source/contract.json`）。提交时检查文件为任务内普通文件，并核传输后的存在、大小与内容；缺失就阻断提交，不把 JSON 改写成 Python 绕过 staging。

- 换目标环境先核 Python SOABI、架构、glibc、libstdc++ 符号版本及 Torch/PTA 与 native wheel/扩展的 ABI 和链接依赖；版本名相同不代表二进制兼容，不兼容时只按授权在私有环境重建。

- 使用选定 venv 的原始绝对 Python 路径执行，resolved 路径只作身份记录；按实际服务的插件过滤、`PYTHONPATH` 和 `--model` 配置上下文加载最终注册类，记录 `classfile`，不实例化或读取 tensor 权重。metadata 成功不能替代类导入成功。

- pip 前先查实际容器的网络模式；`network=none` 时先确认是否有已授权的出口。再用同一执行面、代理和证书检查实际包索引/下载源，API 可达不能证明 pip 可达；无出口先记录阻塞，不反复安装或修改公共网络。

- 失败环境、已完成阶段和原始日志保留；修复后用新 job 从未完成阶段续接，不清空旧环境或重装已完成链路。业务目标改变时另存当前 checkpoint 与新的下一步。

- 给 GLM 的状态查询默认显式加 `--compact`（仅支持 `fleet-job-status`）；完整原始 response 留在绝对 `response_path`，compact 不含 `response`。缺少 `result.json`、响应未到或 JSON 不完整时按证据保持 `unknown`/未完成，禁止盲索引造成 `KeyError` 或凭相对路径猜结果。

- `OPERATOR_UPDATE.json` 已写不等于当前 GLM 已读；必须看到本轮成功的 Read，或实际状态查询工具结果包含对应更新 id 与指示，尤其不能漏掉 stop。未消费时不能宣称目标已切换。

- 从环境准备切换到模型业务等目标变更且 active 未消费时，先核真实请求、活动 job 和 checkpoint；无业务启动才保存断点并按授权核验本 controller 的 owned cleanup，再用 server-use 3.8.8 正式 `run_task.py --resume` 传短任务。已有业务只跟原 job，不重放；不复制 launcher，不接连重启，不将偏离直接归因为超时/API。

按以下边界推进：

1. 环境与加载：核对原始配置、Python/CANN、插件、权重和实际 import 路径。离设备检查先执行目标版本的 `load_general_plugins()`，再对目标架构实际选中的 Registry entry 调用 `load_model_cls()`，保留插件配置、注册覆盖日志、最终类名与 `classfile`。额外 entrypoint 可能覆盖同名模型，`import vllm` 成功不能替代最终类加载；按模型、版本和依赖配置插件 allowlist，对照默认与筛选配置，不全局固定盘古名单。已有 inventory/probe 可辅助核对版本和来源，是否覆盖实际类加载仍须检查。
2. 设备与映射：重新检查健康、空闲、物理卡到逻辑卡映射和本任务占用。动态选卡后记录完整映射；换卡必须重跑对应原生基线组，不把上一张卡的数据移作当前卡证据。
3. 执行链路：先核对冻结的评测、性能和设备 profile 合同，再做真实短生成；当前盘古代表验证使用 48 条评测与 6 组性能。每一阶段独立记录成功、失败、跳过和超时。
4. 通信：区分通信组初始化成功与第一个 TP AllReduce/AllGather 失败。首个 collective 失败时保留 rank、设备映射、日志和初始化结果；根因未确认时不归咎 vendor bug。遇到 AICPU 507018，按 [分层排查与精确跳过条件](references/aicpu-507018.md) 查首个设备 kernel 错误、实际包与成功反例；不据顶层码禁用整个 TP/P/D，也不盲增 timeout 或切 HCCL 模式。
5. Profile：按 [fullnet-profile 离线恢复](../fullnet-profile/references/offline-recovery.md) 先检查所有 worker 的实际 `PROF_*/device_*/data`。有非空设备二进制就优先离线恢复，不重新加载模型。核对真实解析 UID/工具 owner、SDK/Python、依赖、子进程 stderr 与新生成的设备事件；父 exit0 和 CPU trace 都不足以通过。OOM、AICPU 执行失败和解析失败分别归类。把 venv 执行路径和实际 ELF resolved 文件分开记录，核对 symlink 目标但不要把命令替换成 `/usr/bin/python`；SQLite 预检通过也不足以通过解析门，必须用同一命令环境证明完整 selected CANN runtime closure（包括 ABI 核对的 `libc_sec.so`），并以源码片段和子进程负例保留未通过边界。
6. 模型执行或代码修复后的复测创建新实验，保持输入、模板、阈值和分母不变。已有原始数据的离线解析修复创建独立恢复目录，保留旧实验与原件；不要求为解析问题重跑模型。新设备证据经独立检查后再交调用方回填，未恢复时保持 `incomplete`/`pending`。

## 固定诊断顺序与证据边界

新计划按[实验阶段与验收](../../../docs/acceptance-contract.md)执行：**资产与依赖 → 新鲜设备映射、健康与空闲 → 插件加载后的真实模型类导入 → 模型 smoke → P/D 真实 KV 检查（适用时）→ 冻结评测（当前盘古为 48 条）→ 性能与 profile/timeline → cleanup**。每个 case 都在业务 checkpoint 的 `cases` 下记录 `status`、真实 `job`/`request`、`experiment`、`next_action`。恢复先读业务 checkpoint。读取器兼容旧现场的 `four_types`，空 `cases` 会继续读取非空 `four_types`；结构错误直接报错。

### 跨机日志收集与资源续接

新 case 使用 `scripts/ops/cross_host_case.py` 的正式阶段入口；角色本地使用 `scripts/ops/cross_host_role.py`，将同一 actor 文件中的 global rank/world 和加载完成记录发布到当前 job。`world_size=2 rank=1 local_rank=0` 中的 global rank 是 1；`Loading weights took ...` 和 `Model loading took ...` 表示加载完成。收集器使用原始日志回归，单独记录开始加载的文本。

资源变化时，先区分尚未入队的意向和已提交作业。前者撤回意向后回原队列；后者通过正式 requeue 入口核对当前 request/job、模型未开始、资源失败和各角色清理记录。已有活动作业接回原请求，业务首错进入调试流程。

### P/D 请求成功但传输失败

P/D 在 smoke 后先用真实请求检查 KV，再进入固定评测。按同一请求和测量窗口关联两侧传输完成记录、成功计数变化与日志；配置、connector 注册日志、`request_prefill_kv_computed_tokens` 和 HTTP 成功都不能单独证明 transfer completed。消费侧可能在 KV 加载失败后重算并返回 HTTP 200，因此窗口内失败计数增加，或出现 link failure、`fail_pull`、KV 失败后 recompute，均拒收本次传输并停止后续评测；缺少完成证据时保持待验收。

旧 probe 的 passed 与这些失败证据冲突时，保留原始 state/probe，另存独立拒收结论；先修实际调用方的前置检查和测量拒绝门，再按授权复验。文档顺序不代表代码已经自动执行该门，也不证明任何后续 case 已通过设备验收。

设备映射统一使用 `fullnet.device_mapping`：`Chip Phy-ID` 关联 host-bound ACL runtime，`Slot ID`、`Chip ID` 和 `Chip Logic ID` 只作原始字段，不能作为 runtime fallback；ACL 证据必须带实际 host、指定 API、`rc=0` rows 和 `visible_unset=true`，缺失或重复映射即停止。

CPU 诊断在 Python 中使用 `fleet_submit(..., devices=[])`；CLI 使用 `--devices --args-json '["--cpu-only"]'`，即 `--devices` 后零个整数。设备任务显式写 `--devices 1 2 ...`，省略设备合同会在入队前失败。runner 检查设备健康与空闲，broker 校验目标清单与任务路径。新 plan 的模型预检、smoke、P/D 检查和测量由 CLI 按合同串联，GLM 据原始结果定位失败并继续未完成阶段；普通 CPU 诊断使用自己的任务合同。

### 有界首错提取

对项目内实验 service/PD 角色日志和任务日志，使用 `scripts/extract_failure_context.py`，按输入路径稳定遍历并流式读取。目录输入只收集 `.log` 以及明确的 `device*.txt`/`exit*.txt`；显式传入的安全日志文件仍可读，`settings.json`/`*settings*.json`、`.env*`、`.claude` 和 `.codex` 认证配置目录始终跳过。输出为定长 JSON：`first_traceback`、`first_error`、带原始路径与行号的上下文、独立的 `tail_summary`、文件截断和跳过记录。它必须保留更早的 pagefault/AICPU/507018/11003/kernel 错误上下文，不能只取 head/tail 或过滤到最后一条 `ERROR`；输出明确“首见错误是证据，不等于根因”。凭据路径不读取，日志中的 token/password 等值脱敏。

历史 `device-post.txt` 的 PID 只属于历史观察，不能证明当前存活或占卡。清理前先取得新鲜的 PID、启动身份、完整命令、容器、任务归属和设备进程，并逐项交叉核对；不能仅凭旧 PID kill。`userDevId`/logical ID ERROR 也不能单独升级为根因，必须与更早设备错误、实际 runtime 映射和成功反例一起判断。

读取 fleet job status 时，`state/process/failure/result/validation.json` 在 256 KiB 以内必须完整保留原字符串，不能用尾部切片；超过上限要返回可解析的 `truncated` 描述、原始字节数、上限和路径，不能伪装成完整 JSON。`exit.txt`、`workload.log`、`runner.log` 等原有字段继续返回 string，日志只取有界尾部；每个文件的原始字节数、是否截断、片段起始字节和原始路径放在独立 `file_metadata` 表。看到截断对象时先读取原件或按证据边界标记不完整，不要反复另写脚本猜测。

提交与命令退出边界：`fleet/responses/<run-request>.json` 的 `submitted` 只证明请求入队，不是 job 结果；提交后必须另发 status request，并读取对应 `state`、`exit`、`failure` 和结果文件。Bash 管道必须启用 `set -o pipefail`（或等价方式）保留上游退出码；不能用 `head` 截断、`|| true` 或管道末端的 0 把失败伪装成工具成功。有界输出使用现有 helper。

读取真实 fleet job 优先使用同一入口的 `--action fleet-job-status --host <登记IP> --job <job> --wait-seconds 15 --compact`，并传绝对 `--project` 和 `--task`。该操作自动创建 status 请求；已有未返回的同 host/job 查询会复用，返回原件及 `job_outcome`。完整模式 pending 时 `response` 恒为 dict（空对象表示尚未收到结果），不能写成 response 路径或 job queued。带新 schema、完整 `state.json`、`status=exited`、整数 `exit` 且 `file_metadata.state.json.json_valid=true` 的不可变 terminal response 可复用，但它只是历史 job 结果，不是新鲜设备空闲证明；running、查询错误、未完成或旧 schema 必须刷新 status 请求。每次读取（包括缓存命中）都重新读取任务内 `repair/OPERATOR_UPDATE.json`。CLI `fleet-status --request-id <run-request>` 也会在提交成功后自动续查实际 job，并保留 `run_request_id` 和 `submission_response_path`；Python 低层 `fleet_status()` 仍只读指定请求回执。原始 broker JSON 是 raw 对象，CLI 查询结果才包含 `response` 外层，不凭猜测取 `response` 再用空字典掩盖结构不符。无论缓存还是新查询，response_path 都按项目归属、symlink 和大小门控后读取。任务内 `repair/OPERATOR_UPDATE.json` 的新指示通过 CLI 返回，先核对再继续，无须重启活动模型或控制器。

### OPP 路径最小诊断

遇到 `ASCEND_CUSTOM_OPP_PATH` 时，先分别记录变量缺失、精确空字符串和非空路径：空字符串必须与 `unset` 做 A/B 对照；非空值不得在未授权时删除或改写。长项目根或中文 SDK 导出路径与 `TransData`/`IsNameValid` 同时出现时，只能记录为 observed path failure；在同一 SDK、同一 Python 和子进程私有环境中完成对照 A/B 前，根因保持 pending，不把路径相关性写成已证根因。`samefile` 只证明 source 与 ASCII SDK 身份相同，不等于 source 导出的环境已完成 ASCII 路径规范化。A3 已有成功入口的 `a3_cann_exec.py` 时，复用其实际存在的只读 helper，保持 source → 同 SDK 身份校验 → helper 转换 env → selected Python child 的启动顺序；记录 source 后及 helper 转换后的 `ASCEND_HOME_PATH`、`ASCEND_OPP_PATH` 实际导出路径，确认子进程实际收到转换结果。检测成功不能写成修复成功，env 已规范也须等待原失败模型路径复测。案例原件留在任务目录后台维护；公共 README、通用 skill 和公开报告不得写入私人 host、PID 或认证细节。

fleet 回执与连接：提交结果保留绝对 `business_run`/`response_path`；status 的 `job_outcome` 区分 `submitted_not_validated`、`running`、`failed`、`succeeded`，业务验收固定保持 `NOT_VALIDATED`，精度/profile 需另行核验。Bash `cwd` 会继承已有 `cd`，优先传绝对任务路径，避免重复相对 `cd`。broker 使用登记 alias；认证失败先核实际入口，不改密码。`MODEL_COMPLETED` 是正常模型终态，不自动标记为 API 失败。

计算与 GLM 启动：计算任务可以并行，GLM 的启动握手必须串行。一个任务使用 canonical 入口启动后，先核对自身 `SUBMITTED`、代理真实存活或端口绑定归属，以及工具 `Read`/`Bash`（至少确认代理绑定归属），再启动下一个；driver 返回不等于代理就绪。公共入口冷启动可能预选相同代理端口，若 driver 报 `OWN_PROXY_EXITED` 且 proxy stderr 出现 `remote port forwarding failed for listen port`，先查真实远端进程、启动身份、完整命令和是否已有 business run，再判定恢复或重提。只按 UID、startticks、完整命令和任务归属精确 TERM/清理；不盲目重复提交，不停止活跃 profile、其他任务或公共 launcher，不改模型、权限或公共环境。

结果文件由脚本一次写出完整 JSON。诊断只保留必要参数和统计，较大的原始数据单独保存并给路径；不要把 JSON 按字符切进几个文件。fleet runner 会在子进程 exit0 后解析已有 `result.json`、`validation.json`，非法文件使任务失败；紧凑结果文件上限 16 MiB。旧 job 不改写退出码，status 通过 `file_metadata.json_valid/json_error` 和 `job_outcome.artifact_validation` 单列非法产物。出现此错先修写文件的脚本，不拼接猜测、不重跑模型。

Python 任务入口固定遵守退出码边界：主函数正常返回 0 后，在异常处理块之外执行 sys.exit(exit_code)；不得把 sys.exit(main()) 放在 except BaseException 覆盖的 try 内，否则正常的 SystemExit(0) 会被误记为失败。异常路径只捕获真正的 Exception，写出首错并返回非零；入口回归必须同时验证正常返回 0 -> exit 0 和异常 -> 非零，不能只看结果文件。

设备门先于 workload：若 runner 报 `device ... is not confirmed healthy and empty` 且没有 `process.json`，归入 `device_resource_gate`、`workload_execution=not_started`。重新查资源并选择其他健康空闲卡或已登记机器，不把它算成模型执行、通信或精度失败。

修复脚本执行前先做文件写入安全审查：检查 `chmod`/`chown`/删除和临时文件操作是否解析并限制在任务目录内，特别是 `Path.chmod` 等对 symlink 的操作不能把权限变更导向任务外库或公共环境。发现 review-blocked 风险时停在执行前，保留原脚本，按最小修改替换并用唯一 CPU job 复核；只优雅停止和清理本任务持有的 broker/worker，不触碰其他 broker、GLM 或活动 profile。

允许在项目私有目录内自主修复和复测；遇到公共环境、他人任务、权限、驱动或设备 reset 需求立即停止并保留证据。禁止 NPU 带外热复位。
