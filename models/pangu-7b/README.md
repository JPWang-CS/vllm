# Pangu 7B

本项目承载 Pangu 小模型的整网启动、固定样本精度及算子替换实验。A3、A5 的示例配置在 `configs/`；真实权重、运行环境、卡号和启动参数通过本机 `.local.yaml` 明确指定。

运行前按实际模型配置确认层数、dtype、量化方式和平台实现。示例采用 eager 模式用于建立初始基线；图模式作为独立变更对照。

实验按 `experiments/<id>/` 保存。`baselines/index.yaml` 只引用已接受的实验。目前尚未登记本框架在 A3/A5 的整网基线。

评测流程见根目录 `docs/evaluation.md`；AI 执行入口为 `fullnet-validate` skill。

## 权重来源

公开下载入口为 [openPangu Embedded 7B V1.1](https://huggingface.co/openpangu/openPangu-Embedded-7B-V1.1)。该配置为 `PanguEmbeddedForCausalLM`，34 层、hidden size 4096、32 个 Q 头与 8 个 KV 头，BF16 权重。模型与推理实现的版本一起保存到实验配置。

```bash
fullnet assets plan --model pangu-7b
fullnet assets verify-source --model pangu-7b
fullnet assets fetch --model pangu-7b
fullnet assets check --model pangu-7b
```

已有权重按架构、层数、维度、tokenizer 和发布版本核对后只读复用。随机小模型和其它 7B 架构分别建立模型项目。

## 思考模式与固定评测

Embedded 7B V1.1 默认使用慢思考。发布方提供的快思考控制方式是在用户输入末尾加入 `/no_think`；当前 tokenizer 模板没有读取 `thinking` 参数。准备本模型的固定评测数据时，在 Bash 中使用 `--user-suffix $'\n/no_think'`，把控制标记包含在输入 token 数内。服务器验证脚本已经填写这一参数。

原始 RULER 题目、答案、种子和输出预算保留，模型控制后缀单独记录到样本和生成说明。每次实验固定实际发送的输入；模板或控制方式改变后使用新实验，原数据继续保留。
