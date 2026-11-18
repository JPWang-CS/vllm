# 评测协议

一次评测固定模型权重、量化方式、tokenizer、聊天模板、上下文上限、生成参数、数据与评分方法。运行前的配置和输入保存到模型实验目录，后续比较使用保存的副本。

## 已有 Pangu 方法

现有资料提供 RULER 的 `niah_single_1` 与 `vt` 两任务小样：种子 42，上下文长度 4096、8192、16384，每个任务每个长度 8 条，共 48 条；NIAH 输出预算 128，VT 30。使用官方生成器及 `string_match_all`。输入由 `input + answer_prefix` 组成，并按实际模型聊天模板计算 token 数。已有资料还记录了另一批独立重生成的 16K 样本，应分别保存。

这组结果用于检索和变量跟踪测试。它没有覆盖 RULER 全部任务，也没有已确认的统一通过分数。框架保留 `minimum_score: null`，原样报告分项得分。正式阈值确定后以新的评测配置记录。

### 输入与请求

输入为 JSONL，每条包含：

```json
{"id":"niah_single_1/4096/0","task":"niah_single_1","context_limit":4096,"prompt":"完整输入","outputs":["期望答案"],"actual_chat_tokens":3900,"output_tokens":128}
```

`actual_chat_tokens` 必须用当前模型的真实 tokenizer 和模板生成。输入与输出预算之和超限时停止，不截断原文。MTP 或其他配置占用的上下文按该版本实测规则扣除，不把历史某次预留量作为所有模型常量。

服务返回的 `usage.prompt_tokens` 要与输入记录一致。空答、请求失败、缺失样本全部保留在分母内。`finish_reason=length` 单独计数；输出预算保持原协议，不因得分低临时加长。所有结果保存原始响应。

服务就绪后可以运行独立真实生成 smoke。Pangu 92B runner 默认发送短 prompt、最多 8 个 token，单次请求超时 60 秒；请求沿用正式评测的 seed、temperature、top-p、聊天模板参数和 `add_special_tokens`，将请求、原始响应、usage、生成通道和结果写入 `smoke.json`。Smoke 不读取或改写固定 48 条数据，也不要求 `actual_chat_tokens` 与 usage 精确相等。content 为空但 `reasoning_content` 非空时记录为 reasoning 通道生成；两者都为空或请求失败时立即停止正式评测并释放本次服务。

正式评测默认在连续 3 次通信失败（例如连接/HTTP 超时或 HTTP 错误）后停止发出请求；阈值可在 evaluation 配置 `consecutive_communication_failure_threshold` 中调整。每次尝试后把进度写入 `evaluation-progress.json` 和 `state.json`。已发出的请求保留原始记录；阈值触发后，将其余样本写成 `status=skipped` 的未执行记录。结果分别报告 attempted、succeeded、failed、skipped 和 timeout；timeout 是 failed 的子集。得分分母仍是冻结数据集全部样本，协议和评分函数不变。

92B 外层 runner 在成功和失败路径都生成 `validation.json` 与可读 `task-report.md`，并导出报告、状态、评测结果、性能 summary 与 profile artifact 索引。失败时保留非零退出码和错误阶段；如果尚未创建 experiment，也导出 task 级失败报告。性能采集只在正式评测协议通过后运行。

Profile 只有在 trace 中发现带正时长的 Ascend/device kernel 事件，或 kernel CSV 中发现具备设备/任务类型标志且时长大于零的行，才标记通过。CPU trace、空 CSV 和无法解析出设备事件的文件会保留在原路径，并在 `performance/profile-artifacts.json` 中标记 `not_accepted`；导出只保存索引，不复制大型原始 trace。

Pangu 92B runner 的 aligned-decode 环境仅由实验选项决定：默认清除 shell/reference 中继承的 `OMNI_HYBRID_ALIGNED_DECODE`，使用运行时默认；`--disable-aligned-decode` 固定为 `0`，`--enable-aligned-decode` 固定为 `1`。runner 在保存 experiment 前核对最终环境和冻结配置，validation 与 identity 记录最终值。

### 三项结果

| 项目 | 检查对象 | 产物 |
|---|---|---|
| 服务运行 | 服务加载、模型名、真实请求 | service.log、raw/predictions.jsonl |
| 任务精度 | 固定数据和官方评分 | results.json、report.md |
| 数值一致性 | 相同输入的逐层输出或 logits | 导出张量与 parity 结果 |

相同输出文本用于回归检查，数值一致性按原模型或算子的阈值验收。JSON 张量导出格式为 `{"logits":{"shape":[2],"values":[0.1,0.2]}}`；大型张量可由模型适配器使用其原生比较器处理，保留逐项指标。

## 准备和复用数据

`scripts/evaluation/prepare_ruler_subset.py` 支持官方源码、本地 tokenizer 和固定种子。模型与官方源码均从只读目录读取，输出在本项目内。生成参数、实际输入长度与原始样本索引随数据保存。跨硬件比较时复用同一批已经生成的数据，先确认 tokenizer/模板一致。

原有评测方法来自 Pangu 项目保存的 portable evaluation baseline。官方资料：[NVIDIA RULER](https://github.com/NVIDIA/RULER)。
