---
name: fullnet-validate
description: 在指定模型项目中启动推理服务，复用已定评测数据与协议，保存请求、任务精度和清理证据。
---

# 整网运行与精度

适用于“拉起模型”“验证精度”“保存原生基线”。先读目标模型的 `README.md`、`model.yaml`、本机配置和 `docs/evaluation.md`。

1. 确认个人容器、项目真实路径、已登记 CANN/Python、权重格式、平台插件、卡号映射和当前健康空闲状态。环境只读复用；A3 与 A5 分别确认。新用户按 `docs/from-zero.md` 使用对应官方镜像，在项目内执行 `scripts/bootstrap.py`；缺 pip 时使用项目私有 virtualenv 引导。原生基线的 `changes` 为空。
2. 已有权重检查只读路径；首次下载先 `fullnet assets plan`、`verify-source`，再 `fetch`、`check`。公开模型和用户授权模型各用其声明来源。访问失败保留状态，凭据从指定环境变量读取。用户暂不下载权重时可核验下载入口，真实模型步骤仍标记未运行。
3. 找到已有评测数据与评分协议。保持样本 ID、tokenizer/模板、上下文预算、输出预算、种子及评分函数；把协议与数据来源填入配置。缺少阈值时记录任务得分，向用户确认正式验收标准。
4. 执行 `fullnet check --config ...`，然后 `fullnet plan --config ... --run <id>`。核对保存的配置与数据；不以修改样本或缩短输入解决预算错误。
5. 在项目个人 tmux 中执行 `fullnet run --model <model> --run <id>`。按 `docs/background-jobs.md` 提交长任务，AI 的 Bash 调用在提交后返回，业务继续由独立 tmux 监督。该命令接管本次服务进程，检查服务模型名、逐条发送请求、保存原始响应并收尾。上机由获准的工具型 GLM 执行时，保留真实 Read/Bash 事件；独立检查业务退出码及原始结果。
6. 执行 `fullnet report --model <model> --run <id>`；检查请求失败、缺失、截断、评分分母和任务范围。设备释放由容器外部检查补充，不把发送退出信号写成显存释放通过。

多配置部署先通过一项原生基线，再展开 TP/DP、图执行和 MTP 的组合。每个配置使用独立实验目录。设备忙、设备检查失败或上一任务未清理时立即停止后续配置；批处理保留非零退出码及未执行原因。进程编号和设备映射按当前查询确定。

报告同时列出请求成功数、任务得分和截断数。HTTP 请求全部返回时，任务得分仍可能偏低；沿用固定协议保存结果，并另建实验分析模板、生成参数或输出预算。用户明确要求后台保留服务时使用 `--keep-ready`，把实际就绪状态、地址和进程写入记录；其余服务按任务规则收尾。

短生成 smoke 与固定评测分开保存。Pangu 92B 默认使用最多 8 token、60 秒超时的 smoke，并复用正式聊天模板参数；不以 smoke 的 usage 取代固定样本 token 合同。连续通信失败默认阈值为 3，超出后明确写 skipped 和进度，不移除失败样本或调整分母。

正式评测失败、启动 smoke 失败或 profile 被拒绝时，仍需生成 task 报告和 export。性能阶段只在完整固定评测通过后启动。Profile 通过状态必须由 Ascend/device kernel 事件或带设备/任务类型的非空 kernel CSV 支持；CPU trace 不能作为设备 profile 证据。

质量评测、输出对照、逐层数值检查是三项结果。`quality_status=not_specified` 表示尚无已确认的得分阈值。替换算子时还需运行 `fullnet-replace` 的对照流程。

先用 `examples/offline/config.yaml` 验证固定命令和状态记录；该 HTTP fixture 没有模型计算，不能计入设备或精度验收。
