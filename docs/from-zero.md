# 从零准备环境与模型

初次使用依次准备推理环境、框架、模型和数据。项目目录保存下载、缓存、实验与日志；用户提供设备和存储位置。

## 1. 推理环境

可采用官方预构建 vLLM Ascend 镜像，或已匹配版本的个人 Python/CANN 环境。官方镜像按 A3、950DT 分开发布，包含相匹配的 CANN、PyTorch、TorchNPU、vLLM 与 Ascend 插件。宿主驱动按所选发布的要求核对。

选定同一发布版本后，分别使用 `-a3` 和 `-a5` 标签。镜像标签、设备映射、挂载与容器名保存在本机配置中；正式服务器的已有容器不会由框架自动重建。以官方安装页确认当前发布和兼容矩阵：

- [vLLM Ascend 安装](https://docs.vllm.ai/projects/ascend/en/main/getting_started/installation.html)
- [官方预构建镜像](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/getting_started/installation/prebuilt_image.inc.md)

进入个人容器后，手动 source 选定的 CANN，执行：

```bash
python scripts/bootstrap.py --system-site-packages --extras download,evaluation
source envs/fullnet/bin/activate
fullnet doctor
```

框架安装仅添加编排和所选数据工具。使用官方镜像原有的模型依赖；Pangu 的定制模型和量化格式还需其平台实现支持。`doctor` 读取版本元数据，不初始化设备。

## 2. 模型获取

先用公开示例检查下载能力：

```bash
fullnet assets plan --model qwen3-0.6b
fullnet assets verify-source --model qwen3-0.6b
fullnet assets fetch --model qwen3-0.6b
fullnet assets check --model qwen3-0.6b
```

`plan` 输出来源、版本和项目私有目标目录。`fetch` 调用官方 Hub 下载，可从中断处继续；`check` 检查模型配置、tokenizer 与索引中的权重分片。模型注册表 `download` 声明来源仓和版本，凭据通过环境变量传入。

代理网络中的 TLS 证书由本机信任链处理。下载器优先沿用 `REQUESTS_CA_BUNDLE`，也支持 `FULLNET_CA_BUNDLE` 指向用户已信任的 CA 文件；默认采用 Python 的系统 CA 路径。证书校验保持开启，框架不安装或修改公共证书。

Pangu 7B 已登记 openPangu 官方 V1.1 下载源，可把上述命令中的模型名改为 `pangu-7b`。92B 的特定量化资产按发布方授权来源填写；本机已有只读权重则直接在运行配置中指定路径。下载源不可访问时保留失败原因，不用其他模型替代该模型的精度结果。

## 3. 数据与运行

取得评分源码并准备官方 RULER 数据，下面以公开的 Qwen 模型为例：

```bash
python scripts/evaluation/get_ruler.py --output cache/RULER
python scripts/evaluation/prepare_ruler_subset.py \
  --ruler-root cache/RULER --model assets/weights/qwen3-0.6b \
  --output assets/ruler-qwen3 --model-template-token 64 \
  --chat-template-kwargs '{"enable_thinking":false}'
python -c 'import pathlib; p=pathlib.Path("assets/ruler-qwen3"); (p/"all.jsonl").write_text("".join(f.read_text() for f in sorted(p.glob("*-*.jsonl"))))'
fullnet configure --model qwen3-0.6b --platform a3 \
  --python "$(command -v python)" --weights assets/weights/qwen3-0.6b \
  --dataset assets/ruler-qwen3/all.jsonl \
  --official-metrics cache/RULER/scripts/eval/synthetic/constants.py --devices 0
fullnet plan --config models/qwen3-0.6b/configs/a3.local.yaml --run a3-native-001
fullnet run --model qwen3-0.6b --run a3-native-001
fullnet report --model qwen3-0.6b --run a3-native-001
```

A5 把平台改为 `a5`，使用自己的设备号和 A5 环境。框架运行前核对所选引擎支持的参数与上下文上限。A3/A5 使用相同模型、模板和生成协议；运行配置填写实际环境版本与设备信息。

AI 使用 `fullnet-validate` 按检查、计划、运行、报告的顺序执行。新增模型时建立自己的 `model.yaml`、配置和实验目录，复用公共工具与 skill。
