---
name: fullnet-validate
description: 在指定模型项目中启动推理服务，复用已定评测数据与协议，保存请求、任务精度和清理证据。
---

# 整网运行与精度

适用于“拉起模型”“验证精度”“保存原生基线”。先读目标模型的 `README.md`、`model.yaml`、本机配置和 `docs/evaluation.md`。

## 目录归属

模型、配置、变更、基线和实验遵循 [模型与实验目录规范](../../../docs/repository-layout.md)。使用 `models/<model>/configs/` 保存该模型配置，使用 `models/<model>/changes/<change-id>/` 保存受控变更，使用 `models/<model>/experiments/<run-id>/` 保存冻结配置、数据、日志、原始输出和报告。原生 baseline 使用一个实验，替换 candidate 使用另一个实验；每个实验内按合同保存精度、性能和 profile，输入、配置或实现合同改变时才创建新 run。不要跨模型复用本地配置或实验结果。

1. 先记录原始配置和实际加载路径：个人容器、Python/CANN、平台插件、权重、卡号映射及 A3/A5。健康检查同时确认物理卡与运行时逻辑卡的映射；动态选择空闲卡不等于只选一张卡。换卡后，原生基线、固定评测、性能和 profile 必须整组重测。
2. 权重首次使用先执行 `fullnet assets plan`、`verify-source`、`fetch`、`check`；已有权重只读核对架构、分片、tokenizer 和来源。环境变量提供凭据，不把凭据写入配置或日志。
3. 模型 runtime 的源码、Python package data 和 native extension 必须完整进入实际执行源；不要按目录名一律排除 `cache`，其中可能是真实 Python module。数据缓存可以排除，但必须与 package、模板、静态资源、pkgdata 和扩展库分开判断。源清单发现内部 module、package data 或 native extension 缺失时，先停在 source gate，不重复加载模型。
4. 在加载权重前，用实际目标 Python、实际 `sys.prefix` 和实际 `PYTHONPATH` 做离设备的依赖、服务入口和真实 router module import 检查。可从真实 `serve` package 的导入关系生成有界检查；不要用不完整的 FastAPI stub 调用 router attach，也不要把 `--help` 或静态存在性当作 import 通过。需要可能初始化设备的真实服务 import 时，先通过 fresh 健康/空闲卡门。
5. 冻结样本 ID、模板、输入/输出预算、种子、评分函数和并行配置。缺少质量阈值时记录得分和 `quality_status=not_specified`，不把分数改写成质量通过。
6. 依次执行 `fullnet check --config ...` 与 `fullnet plan --config ... --run <id>`，检查冻结配置和数据。不要通过删样本、缩短输入或改变生成参数绕过预算错误。
7. 可选 `quality-only` 必须同时传播到启动 argv、配置和阶段状态；它只把 profile/performance 记为 `deferred_by_user`，不改变默认 full-chain 合同，也不删除默认 profile/performance 门。未显式启用时仍执行原完整流程。
8. 短生成 smoke、固定评测、服务性能和设备 profile 分阶段保存。先用真实短生成确认服务可用，再进入固定评测；性能与 profile 只在评测阶段满足前置条件后开始。
9. 前台或后台任务都使用项目私有 tmux。后台提交使用仓库已有的 `scripts/ops/submit_job.py`，完成后读取 `job-state.json`、`job-exit.txt`、`results.json` 和清理记录；工具会话退出、业务退出和模型验收是三件事。
10. 使用 `fullnet report --model <model> --run <id>` 汇总结果，同时分别记录 raw 48、attempted/succeeded/failed/skipped/timeout、截断、评分分母与 score、质量阈值状态、业务退出码和 cleanup。截断结果不能写成完整 48 条预算通过；发送终止信号不能单独证明显存和端口已经释放。

样本数、评分和性能负载按模型的冻结合同执行。当前盘古代表验证采用 48 条评测与 6 组服务性能；其他模型使用各自登记的协议。

A3 与 A5 独立验收。`examples/offline/config.yaml` 和 A3/A5 HTTP fixture 只验证编排、请求、评分和状态记录，不包含模型推理，不能计入真实设备或精度验收。

正式验收必须保留首错和逐阶段状态。质量、输出对照、逐层数值检查、服务性能和设备 profile 分开记录；失败时仍导出 task 报告和原始结果，不覆盖旧实验。禁止 NPU 带外 reset、驱动变更和共享环境修改。
