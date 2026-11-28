---
name: fullnet-validate
description: 在指定模型项目中启动推理服务，复用已定评测数据与协议，保存请求、任务精度和清理证据。
---

# 整网运行与精度

适用于“拉起模型”“验证精度”“保存原生基线”。先读目标模型的 `README.md`、`model.yaml`、本机配置和 `docs/evaluation.md`。

## 入口路径与 cwd

GLM 或任务脚本的起始 `cwd` 可能是 controller/task 目录，不要把它当作项目根。执行命令前固定做三步：

1. 从 `TASK_CONTEXT` 和 `RUN_IDENTITY` 读取并记录实际 `project`、`business_run`、`controller_run`；用 Bash 先核对 `pwd`，再用这些绝对路径定位文件。源码和脚本使用 `Path(project)/scripts`，业务 spec、结果和输入使用 `Path(business_run)`，控制器回执使用 `Path(controller_run)`。
2. 命令、spec、配置和脚本都传显式绝对路径；不要凭 `./`、当前 cwd 或 `repair/...` 猜项目根，也不要把 controller/task 目录中的同名文件当作 business 文件。
3. 文件不存在时，先按绝对路径逐层检查父目录、真实路径和目录归属，再判断是路径错误还是尚未上传；不得从其他项目复制同名文件，也不得通过改权限掩盖路径或挂载问题。

## 运行前与续接检查

- 跨目标复用 runtime 前，核 Python SOABI/架构、glibc、libstdc++ 符号和 native wheel/扩展的 Torch/PTA ABI、链接依赖；不兼容二进制不能直接搬用，不升级系统库来迁就它。

- 用 selected venv 的原始绝对 executable，保留 `sys.prefix`、实际源码、插件过滤与最终 `classfile`。按服务真实 `--model` 上下文执行下节类导入；仅查包 metadata 或 wheel 存在不足以放行。

- 安装前确认实际容器网络模式；`network=none` 先核已授权出口，再检查同一执行面到实际 pip 索引/下载源的连通性。控制面或模型 API 可达不代表 pip 可达；禁止盲重试或擅改公共网络。

- 环境准备失败保留私有环境、阶段结果和首错；修复后新建阶段 job 续接未完成部分，旧 job 原件不改，不重装已完成链路，不将安装成功算作模型验证。

- GLM 查询 fleet job 默认显式使用 `--compact`，完整 response 通过绝对 `response_path` 读取；compact 没有 `response` 外层。先看 `job_outcome` 与文件完整性，缺结果保持 `unknown`/未完成，不盲取 `result.json` 或猜相对路径。

- 提交、远端创建与业务完成分三步核对：`fleet-submit` 的 `submitted_to_broker_queue` 表示请求已保存；该请求的响应中 `submitted=true` 表示远端作业已创建；再通过 `fleet-job-status` 读取执行结果。等待前两步时保留同一请求，续接先比对 host、job、script、devices、args，再接回原请求。完整状态文件中的退出码为 `state.json.exit`；使用 `job_outcome` 或解析 JSON，不根据截断文本猜测结束条件。

- CPU 准备使用最终启动参数，包括尝试编号与角色后缀。Ray 路径交给当前适配器检查完整名称、项目归属和 socket 长度。检查计划、依赖、环境、设备映射与评测合同齐全后再放行；`CPU_PREPARED` 或 exit0 本身不代表计划可启动。

- 目标项目正在运行且需要更新源码时，broker 保留原请求，以 `fleet/waiting-<request>.json` 的 `WAITING_TARGET_PROJECT` 状态继续检查，不返回部署失败、不加载模型、不重复提交。若目标发布文件逐文件内容一致，broker 只读核对并复用它们，将新脚本、输入和结果写到各自作业目录；`framework-sync.json` 记录 `reuse_identical_active` 和未写源码的证据。未知错误仍按失败处理。
- 目标项目空闲时同样先只读比较 published source；全量一致记录 `reuse_identical_idle` 并只创建本次 job。pending inventory 在兼容且新鲜 heartbeat 下复用同一 query 等待，不要把 broker staging 等待误报为 model timeout；heartbeat 失鲜、响应错误和真实模型超时分别记录。

- 新任务指示写入 `OPERATOR_UPDATE.json` 后，核当前执行实际 Read 或状态查询的成功工具结果已包含对应更新 id/指示；文件落盘不代表已消费，stop 更新同样必须保留并执行。

- 目标切换而 active 尚未消费时，先核 checkpoint、真实请求与活动 job。无业务启动才按授权核清本 controller 的 owned cleanup，再沿 server-use 3.8.8 正式 `run_task.py --resume` 交短任务；已有业务继续跟原 job，禁止重放、复制 launcher、接连重启或无证据归因超时/API。

## 目录归属

模型、配置、变更、基线和实验遵循 [模型与实验目录规范](../../../docs/repository-layout.md)。使用 `models/<model>/configs/` 保存该模型配置，使用 `models/<model>/changes/<change-id>/` 保存受控变更，使用 `models/<model>/experiments/<run-id>/` 保存冻结配置、数据、日志、原始输出和报告。原生 baseline 使用一个实验，替换 candidate 使用另一个实验；每个实验内按合同保存精度、性能和 profile，输入、配置或实现合同改变时才创建新 run。不要跨模型复用本地配置或实验结果。

1. 先记录原始配置和实际加载路径：个人容器、Python/CANN、平台插件、权重、卡号映射及 A3/A5。健康检查同时确认物理卡与运行时逻辑卡的映射；动态选择空闲卡不等于只选一张卡。换卡后，原生基线、固定评测、性能和 profile 必须整组重测。
- fleet inventory 必须在与 runner 相同的目标容器内用同一份 host-matched ELF 查询，并记录 `execution_surface=container`；host board、container board、ChipPhy 和 runtime ID 分开保存，查询失败不得回退 host，最终 lease/队列选择须把新鲜 container board mapping 与实际 ACL 证据合并核对。
2. 权重首次使用先执行 `fullnet assets plan`、`verify-source`、`fetch`、`check`；已有权重只读核对架构、分片、tokenizer 和来源。环境变量提供凭据，不把凭据写入配置或日志。
3. 模型 runtime 的源码、Python package data 和 native extension 必须完整进入实际执行源；不要按目录名一律排除 `cache`，其中可能是真实 Python module。数据缓存可以排除，但必须与 package、模板、静态资源、pkgdata 和扩展库分开判断。源清单发现内部 module、package data 或 native extension 缺失时，先停在 source gate，不重复加载模型。
4. 在加载权重前，用实际目标 Python、`sys.prefix`、`PYTHONPATH` 和服务相同的插件配置检查依赖、服务入口及真实 router module。模型导入按下节检查插件加载后的实际注册类；`import vllm`、`--help` 或文件存在均不足以放行。需要可能初始化设备的真实服务 import 时，先通过 fresh 健康/空闲卡门。
5. 冻结样本 ID、模板、输入/输出预算、种子、评分函数和并行配置。缺少质量阈值时记录得分和 `quality_status=not_specified`，不把分数改写成质量通过。
6. 依次执行 `fullnet check --config ...` 与 `fullnet plan --config ... --run <id>`，检查冻结配置和数据。不要通过删样本、缩短输入或改变生成参数绕过预算错误。
7. 可选 `quality-only` 必须同时传播到启动 argv、配置和阶段状态；它只把 profile/performance 记为 `deferred_by_user`，不改变默认 full-chain 合同，也不删除默认 profile/performance 门。未显式启用时仍执行原完整流程。
8. 短生成 smoke、固定评测、服务性能和设备 profile 分阶段保存。P/D 部署在 smoke 后先通过真实 KV 传输检查，再进入固定评测；普通部署在 smoke 后进入固定评测。性能与 profile 只在评测阶段满足前置条件后开始。
9. 前台或后台任务都使用项目私有 tmux。后台提交使用仓库已有的 `scripts/ops/submit_job.py`，完成后读取 `job-state.json`、`job-exit.txt`、`results.json` 和清理记录；工具会话退出、业务退出和模型验收是三件事。
10. 使用 `fullnet report --model <model> --run <id>` 汇总结果，同时分别记录 raw 48、attempted/succeeded/failed/skipped/timeout、截断、评分分母与 score、质量阈值状态、业务退出码和 cleanup。截断结果不能写成完整 48 条预算通过；发送终止信号不能单独证明显存和端口已经释放。

## 模型注册与 P/D 前置检查

1. 离设备模型导入检查先执行目标版本的 `load_general_plugins()`，再对目标架构实际选中的 Registry entry 调用 `load_model_cls()`。保留插件配置、加载与覆盖日志、最终类名和 `classfile`（例如用 `inspect.getfile` 获取），核对类是否来自预期实现。额外 entrypoint 可能覆盖同名模型，基础包导入成功不代表最终模型类可加载。插件 allowlist 按模型、运行时版本和依赖配置，记录默认配置与筛选配置的结果；不把某个盘古插件名单固定为全局默认。筛选改变后重新检查服务使用的实际注册类。
2. P/D 在 smoke 之后用真实请求检查 KV 传输，以同一请求和测量窗口关联两侧传输完成记录、成功计数变化及日志，再放行冻结评测（当前盘古为 48 条）。配置已启用、connector 注册日志、`request_prefill_kv_computed_tokens` 和 HTTP 成功仅是辅助信息，不能单独证明 transfer completed。窗口内失败计数增加，或出现 link failure、`fail_pull`、KV 加载失败后的 recompute，均拒收本次传输并停止后续评测；证据不足时保持待验收。旧 probe 即使标为 passed，也须保留原件并另存拒收结论。这是执行与验收要求，先核对调用方实际阶段顺序，不据本文假定前置门已实现或已设备验证。

## 并行任务与启动握手

模型、任务目录、端口和输出目录必须分别分配并记录；不同任务不得复用同一可写输出或运行资源。源码/依赖上传使用原子 staging，未完成的副本不能进入启动流程。

跨机作业使用 `repair/worker.dependencies.json` 为 `repair/worker.py` 声明非代码输入，或使用 `repair/staging-manifest.json`。清单名不带 `.py`，内容为 `{"files":["repair/data/config.json","repair/data/all.jsonl"]}`。启动前检查实际作业的 `staging-dependencies.json`：配置、评测数据、两端环境和设备映射证据均已列出并验证。先在实际作业目录完成 CPU prepare，再启动设备阶段；本地清单检查不能代替远端文件到位检查。

物理设备 lease 必须先于 fresh health/mapping 查询取得，并一直持有到 workload 结束和 cleanup 完成。遇到 `lease_busy` 时记录为 `device_resource_gate`、`workload_execution=not_started`；等待当前项目持有者释放或改选其他确认空闲设备，不删除锁文件、不 kill 持有者，也不把它算作模型执行失败。

计算 workload 可以并行；cold launcher 的启动握手必须串行。每个任务先确认自己的 proxy 已真实存活或绑定归属，再允许下一个 launcher 启动；确认后，模型/评测等业务阶段才可按任务合同并行。driver 返回、入队或 `SUBMITTED` 本身都不代表 proxy 已就绪。

## 固定恢复/验收顺序

对每个 case 固定执行：**资料齐全 → 插件加载后的真实模型类导入 → 新鲜物理/逻辑/进程内设备映射及健康空闲 → 模型 smoke → P/D 真实 KV 检查（适用时）→ 冻结样本精度（当前盘古为 48 条）→ profile/timeline → cleanup**。阶段之间不得用旧报告、heartbeat、历史 `device-post.txt` 或父进程 exit 0 代替当前证据。

每个 case 的业务 checkpoint 使用稳定的顶层 `cases` 映射；每项至少记录 `status`、真实 `job`/`request`、`experiment` 和 `next_action`。恢复优先读取业务 checkpoint，不以 controller/broker heartbeat 推断业务已结束。读取器兼容旧或模型生成的顶层 `four_types`，但空 `cases` 不得遮蔽非空 `four_types`；两者均不存在或结构非法时必须显示错误。不要再引入其他 checkpoint 形态。

以下边界已有代码检查，模型与设备步骤仍须逐项执行并留证：

- `recover_framework.py` 的 Python 函数调用可用 `devices=[]` 表示无设备任务；CLI 不要写 `--devices []`（`[]` 不是整数，会被 argparse 拒绝）。CPU 诊断照抄：`python3 -B scripts/ops/recover_framework.py --project <project> --task <task> --action fleet-submit --request-id <id> --fleet-action run --host <host> --job <job> --fleet-script repair/<probe>.py --devices --args-json '["--cpu-only"]'`；设备任务写成 `--devices 1 2 ...`；省略整个 `--devices` 必须在写队列前失败。请求文件采用任务目录内的原子写。fleet runner 会在设备任务执行前检查健康和空闲，broker 会检查 protected devices；业务状态读取与 controller/broker 心跳分离。
- 真实模型类导入、映射与健康/空闲复核、smoke、适用的 P/D KV 检查、48 条精度、profile/timeline、清理核验、阈值记录和 checkpoint 写入仍需执行者按上述顺序完成。质量阈值未指定时记录 `quality_status=not_specified`，不得修改样本或阈值来制造通过。

`fleet/responses/<run-request>.json` 中的 `submitted=true` 表示远端作业已创建；必须另发 status request，再读取该 job 的 `state`、`exit`、`failure` 与结果文件，才能判断阶段结果。Bash 使用管道时启用 `pipefail`，禁止用 `head`、`|| true` 或末端命令的 0 掩盖上游失败；有界输出沿用现有 helper。

对 fleet job，优先调用 `recover_framework.py --project <绝对项目根> --task <任务名> --action fleet-job-status --host <登记IP> --job <job> --wait-seconds 15 --compact`。它负责真正的 status 请求、未返回查询的复用和原始结果读取；完整模式 pending 的 `response` 恒为 dict，合法新 schema 的不可变 exited terminal 可复用但不代表新鲜设备状态，每次读取都会刷新 `operator_update`；running、查询错误、未完成和旧 schema 继续发新 status 请求。response_path 读取仍受项目归属、symlink 和大小门保护。不再由 agent 拼多个请求或猜测工作目录。执行成功仅表示进程退出为 0，评测完整性、质量和 profile 继续逐项验收。具体状态与失败读取规则见 `fullnet-debug`。

每个 `result.json`、`validation.json` 独立保存完整 JSON，不能切成字符片段。runner 在子进程 exit0 后执行结构检查；不合法的结果会使 job 失败。设备门未通过、workload 未开始时选择其他健康空闲资源，继续保留该次检查结果，不重复使用旧卡号等待。

fleet 接口结果边界：`fleet-submit` 回执必须保留 `business_run` 和绝对 `response_path`；`fleet-status` 的 `job_outcome` 按实际证据区分 `submitted_not_validated`、`running`、`failed`、`succeeded`、`unknown` 或 `broker_failed`，`business_acceptance` 始终为 `NOT_VALIDATED`，直到独立完成精度/profile 验收。Bash 的 `cwd` 会保留 `cd` 变化，任务路径优先使用绝对路径，不要在已有绝对 cwd 下重复拼相对 `cd`。broker 连接使用登记 alias；认证失败先核对实际入口，不改密码。正常 `MODEL_COMPLETED` 不归类为 API 失败。

清理前必须用新鲜现场同时核对 PID、启动身份、完整命令、容器、任务归属和设备进程；历史 `device-post.txt` 中的 PID 只能说明过去观察到过该进程，不能证明现在存活或占卡，也不能成为仅凭旧 PID 执行 kill 的依据。`userDevId`/logical ID 映射错误不得单独认定根因；须保留并对照更早的 device error（包括 pagefault、AICPU、kernel launch/lookup 等）和成功反例。

样本数、评分和性能负载按模型的冻结合同执行。当前盘古代表验证采用 48 条评测与 6 组服务性能；其他模型使用各自登记的协议。

A3 与 A5 独立验收。`examples/offline/config.yaml` 和 A3/A5 HTTP fixture 只验证编排、请求、评分和状态记录，不包含模型推理，不能计入真实设备或精度验收。

正式验收必须保留首错和逐阶段状态。质量、输出对照、逐层数值检查、服务性能和设备 profile 分开记录；失败时仍导出 task 报告和原始结果，不覆盖旧实验。禁止 NPU 带外 reset、驱动变更和共享环境修改。

新 run 前先同步已发布 main。源码副本须有项目内 `.runtime/framework-source.json` 与实际已发布快照，broker 按内容检查来源并记录目标同步结果。旧内存 broker 先更新；目标有活动任务时，源码需要更新则等待，源码完全一致则只读复用。保留已有任务，不同步 CANN、模型权重和私有环境。
