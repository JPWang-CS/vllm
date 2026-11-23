---
name: fullnet-validate
description: 在指定模型项目中启动推理服务，复用已定评测数据与协议，保存请求、任务精度和清理证据。
---

# 整网运行与精度

适用于“拉起模型”“验证精度”“保存原生基线”。先读目标模型的 `README.md`、`model.yaml`、本机配置和 `docs/evaluation.md`。

1. 先记录原始配置和实际加载路径：个人容器、Python/CANN、平台插件、权重、卡号映射及 A3/A5。健康检查同时确认物理卡与运行时逻辑卡的映射；动态选择空闲卡不等于只选一张卡。换卡后，原生基线、固定评测、性能和 profile 必须整组重测。
2. 权重首次使用先执行 `fullnet assets plan`、`verify-source`、`fetch`、`check`；已有权重只读核对架构、分片、tokenizer 和来源。环境变量提供凭据，不把凭据写入配置或日志。
3. 冻结样本 ID、模板、输入/输出预算、种子、评分函数和并行配置。缺少质量阈值时记录得分和 `quality_status=not_specified`，不把分数改写成质量通过。
4. 依次执行 `fullnet check --config ...` 与 `fullnet plan --config ... --run <id>`，检查冻结配置和数据。不要通过删样本、缩短输入或改变生成参数绕过预算错误。
5. 短生成 smoke、固定评测、服务性能和设备 profile 分阶段保存。先用真实短生成确认服务可用，再进入固定评测；性能与 profile 只在评测阶段满足前置条件后开始。
6. 前台或后台任务都使用项目私有 tmux。后台提交使用仓库已有的 `scripts/ops/submit_job.py`，完成后读取 `job-state.json`、`job-exit.txt`、`results.json` 和清理记录；工具会话退出、业务退出和模型验收是三件事。
7. 使用 `fullnet report --model <model> --run <id>` 汇总结果，同时检查成功、失败、跳过、超时、截断、评分分母和设备释放。发送终止信号不能单独证明显存和端口已经释放。

样本数、评分和性能负载按模型的冻结合同执行。当前盘古代表验证采用 48 条评测与 6 组服务性能；其他模型使用各自登记的协议。

A3 与 A5 独立验收。`examples/offline/config.yaml` 和 A3/A5 HTTP fixture 只验证编排、请求、评分和状态记录，不包含模型推理，不能计入真实设备或精度验收。

正式验收必须保留首错和逐阶段状态。质量、输出对照、逐层数值检查、服务性能和设备 profile 分开记录；失败时仍导出 task 报告和原始结果，不覆盖旧实验。禁止 NPU 带外 reset、驱动变更和共享环境修改。
