# CPU-only trace 的离线诊断与恢复

先检查旧任务的真实进程、采集窗和输出，保存原始 `PROF_*`。模型已经完成请求时，优先恢复已有数据；只有确认设备原始采集缺失，才提出补采建议。

## 先判断问题在哪一层

| 现场证据 | 判断 | 下一步 |
| --- | --- | --- |
| 只有客户端 `.pt.trace`，没有 worker 的 `_ascend_pt/PROF_*` | 服务侧设备采集尚未确认 | 查服务实际 profiler 开关、worker 环境、采集窗和输出目录；先不重复加载模型 |
| `PROF_*/device_*/data` 有非空 `stars_soc`、`ffts` 等二进制，trace 只有 `cpu_op/queue` | 已有设备原始数据，导出链未完成 | 在独立任务复制实际 `PROF_*`，离线 import、summary、timeline |
| 原始 summary/timeline 有设备事件，框架索引为空 | 索引或过滤遗漏 | 核对索引扫描目录、文件名、字段、设备与 rank；保留原文件，给调用方新的证据索引 |
| 权重加载 OOM、AICPU 执行错误、请求失败 | 模型运行问题 | 按 `fullnet-debug` 定位首错，与导出失败分别记录 |

CPU trace 有 `kernel` 字样也可能只是 Host 提交事件。只有 `ph=X`、正 duration，且属于真实 AI Core/AI Vector 设备 lane 或明确计算 Task Type，才计入设备计算。通信和 AICPU 单独统计。文件存在、父进程 exit 0、`Task execution completed` 都不足以验收。

## 先检查身份、工具与依赖

在实际解析容器内记录 `id`、Python 路径、CANN 实际路径，以及 `msprof` 和 `msprof.py` 的真实路径与所有者。设备运行身份与离线解析身份可能不同：某些 SDK 对工具所有者做校验，使用登记的工具所有者身份解析；不改共享 CANN 所有者或权限。

`msprof` ELF 即使 `--help` 也可能初始化驱动。先只读检查 SDK 的 Python 入口与源码。本工具使用官方 Python CLI，先探测根命令及 `import`、`export summary`、`export timeline` 的帮助；不支持 `--collection-dir` 的版本会报 `BLOCKED_CLI_FAMILY`。此时按当前 SDK 的实际帮助/源码选旧命令族，保存完整 argv，不自动换用一个历史 ELF 命令。

解析进程必须真正接收构造的环境。新 shell 手动 source 当前选定的 CANN，再通过当前任务的 `HOME`、`TMPDIR`、缓存、`LD_LIBRARY_PATH` 运行解析。禁止仅创建 Python `env` 字典而不传给 `subprocess.run`。

导出在调用官方 CLI 的任何 `--help`、`import` 或复制原始 `PROF_*` 之前，必须用 `args.python -B -X utf8` 和将实际传给 CLI 的同一个子进程环境执行纯 CPU `import sqlite3`。收据 `parser-dependency-preflight.json/.stdout/.stderr` 必须记录命令、退出码、Python/SQLite 版本、`LD_LIBRARY_PATH` 以及 stdout/stderr；失败阶段固定为 `parser_dependency_preflight`，并记录首个错误。门失败时 `export` 非零、`RECOVERY.json.status=FAIL`，官方 import 和原始 PROF 复制均不运行。

若子进程报 `ImportError: libsqlite3.so`，注意 `libsqlite3.so`（链接器开发名）与 `libsqlite3.so.0` / `libsqlite3.so.0.8.6`（版本化运行库名）不是同一个文件名。先检查本容器已安装库的实际位置和 ELF 架构；GLM 选定匹配库后，只在新恢复目录建立任务私有 `parser-libs/libsqlite3.so` 链接或副本，将该目录前插到这次导出进程的 `LD_LIBRARY_PATH`，再用上述同一 Python/环境实际导入验证。不能因为版本化文件存在就宣称未版本名依赖已满足，也不能盲配 ABI、安装公共包或修改共享 SDK/权限。保留修复前失败收据，在新的恢复目录重做；不改原始 `PROF_*`、旧 state 或旧 validation。

## 可执行入口

使用当前 skill 的 `scripts/profile_recovery.py`，所有参数填现场真实路径：

```sh
python -B "$SKILL/scripts/profile_recovery.py" inspect "$CAPTURE" \
  --output "$TASK/inspection.json"

# 在登记的离线解析身份、新进程及任务私有 tmux 中执行。
# OUTPUT 必须不存在，且位于当前业务项目内、原件目录之外。
python -B "$SKILL/scripts/profile_recovery.py" export \
  --source "$CAPTURE" --output "$TASK/export-attempt-01" \
  --cann-root "$CANN" --python "$PARSER_PYTHON" \
  --library-dir "$TASK/parser-libs"

# 导出结束后，按采集合同核完整 rank 和设备。参数来自本次任务，以下仅示例。
python -B "$SKILL/scripts/profile_recovery.py" verify "$TASK/export-attempt-01" \
  --expected-profs 2 --expected-devices 0 1 --output "$TASK/FINAL_ACCEPTANCE.json"
```

`inspect` 是只读诊断，exit 0 只表示诊断运行成功；看 `classification` 和 `acceptance`。`export` 复制每个实际 `PROF_*`，保留源路径、原件文件大小/mtime 前后清单，逐 rank 执行官方 import、summary CSV 和 timeline。每一步保留 argv、stdout、stderr、真实退出及子进程错误。超时保留已经产生的日志。旧副本内预存的导出文件不能满足新导出的验收。

已经完成三步官方导出时，只运行 `verify`，输出新的验收文件并保留旧失败；不重新导出。框架 `_offline_parse_profiles` 会复用已有恢复收据，修复解析环境后若需新导出，调用方必须明确给出新的私有 output。证据路径统一以收据中的 PROF copy 为基准；发现的 PROF 数与通过数分别记录，不能把验收失败写成未发现数据。官方工具 UID 与当前进程不同只作为身份记录，是否拒绝以实际 SDK 首错为准，不改公共所有者。

官方 Python CLI 不支持时，可在独立副本用当前 `torch_npu.profiler.profiler.analyse`；该入口接受 worker `_ascend_pt`，不是任意上级目录。`analyse(max_process_number=...)`、`tensorboard_trace_handler(analyse_flag=..., async_mode=...)` 与 CLI 的参数分开核对。异步 daemon 未完成时先查真实子进程和日志，不再次提交同一解析。旧版 `msprof --parse/--export/--output` 只在现场确认版本、身份和入口安全后使用。

进程限制与异步等待分别判断。实际日志若出现 `daemonic processes are not allowed to have children`，表示解析所在的 Python daemon 上下文不能继续创建子进程；保留 raw，在独立、非 daemon 的离线进程解析副本，不修改模型 worker 的 daemon 属性，也不重采。只有异步解析尚未完成而没有这类首错时，才继续核对原解析 PID、日志和终态。本次没有取得该 daemon 限制的原始失败日志，这是一条按明确错误文本启用的诊断路径。

## 完成条件与回填

`RECOVERY.json` 检查已发现的每个 PROF 三步解析、子进程失败、新生成的 summary/timeline 计算事件与原件 metadata。最终还必须运行 `verify`，显式给出预期 PROF 数和设备集合，检查整份 rank 丢失，以及每份 raw、CSV 与 timeline 的设备一致。无显式 Device_id 的 timeline 仅在单设备 PROF 内按来源绑定，并在结果中记录 `source_device_scope`；多设备来源无法绑定时保持未确认。源文件大小/mtime 清单是本轮未改写检查，不宣称字节级完整性证明。

验收还要从原始 CSV 与 timeline 独立复算事件数、设备、Task Type 和 duration，并核对采集窗、rank 映射、预期业务算子及缺失 rank。不同导出表可能过滤了不同任务，不能强求总行数无条件相等。时间线中的 kernel 级任务能说明设备执行和重叠；算子内部 MTE/CUBE/VECTOR 指令流水需要单独的 msprof op/MindStudio PipeTimeline 采集，普通整网 timeline 无法补出该信息。

离线恢复不重写旧实验 state/validation，不改变原精度结论，也不补足未测的性能矩阵。将新证据索引交调用方；调用方复用恢复目录中所有 rank 的 CSV 和设备 timeline，再重新运行原有设备证据检查。活跃 GLM 正在写框架代码时，只提供接口与差异建议，不并发改源码。

回归入口：`python -B -m unittest discover -s "$SKILL/scripts" -p test_profile_recovery.py -v`。测试覆盖 CPU-only、设备 raw 与导出缺失区分、AICPU/通信线程过滤、父 exit0 子 ImportError、超时日志、旧导出假通过、整 rank 缺失与 CSV/timeline 设备错配。超时后需按本任务精确身份检查解析子进程是否退出；脚本超时收据不等于资源清理已完成。
