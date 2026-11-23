# 从零准备环境与模型

初次使用依次准备推理环境、Fullnet Lab 框架、模型、数据和项目私有配置。项目目录保存下载、缓存、实验和日志；CANN、Python、权重和设备按实际环境核对，A3 与 A5 分开验收。

## 1. 推理环境与框架

可采用匹配版本的官方 vLLM Ascend 镜像，或已有的个人 Python/CANN 环境。官方镜像按 A3、A5 分开发布，进入个人容器后手动 source 选定的 CANN，再在项目目录安装框架：

```bash
python scripts/bootstrap.py --system-site-packages --extras download,evaluation
source envs/fullnet/bin/activate
fullnet doctor
```

框架安装只负责编排和数据工具，不升级模型环境中的 PyTorch、torch_npu、vLLM、平台插件或 CANN。`doctor` 读取版本和环境元数据，不初始化设备。安装与兼容关系以实际发布页为准：

- [vLLM Ascend 安装](https://docs.vllm.ai/projects/ascend/en/main/getting_started/installation.html)
- [官方预构建镜像](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/getting_started/installation/prebuilt_image.inc.md)

## 2. 模型来源与权重

先检查公开示例的来源和文件，再决定是否下载：

```bash
fullnet assets plan --model qwen3-0.6b
fullnet assets verify-source --model qwen3-0.6b
fullnet assets fetch --model qwen3-0.6b
fullnet assets check --model qwen3-0.6b
```

`plan` 输出来源、版本和项目私有目标目录；`verify-source` 检查远端清单；`fetch` 下载到项目缓存和权重目录；`check` 检查配置、tokenizer 和权重分片。下载凭据通过模型声明的环境变量传入，证书使用当前信任链或已授权的 `FULLNET_CA_BUNDLE`，不修改公共证书。

Pangu 7B 的公开入口见 `models/pangu-7b/README.md`。92B 等授权资产按实际来源填写；已有权重只读核对架构、量化、分片、tokenizer 和发布版本。来源不可访问或文件不完整时保留失败状态，不用其他模型替代该模型结果。

## 3. 数据与项目配置

取得官方 RULER 资料并准备固定子集：

```bash
python scripts/evaluation/get_ruler.py --output cache/RULER
python scripts/evaluation/prepare_ruler_subset.py \
  --ruler-root cache/RULER --model assets/weights/qwen3-0.6b \
  --output assets/ruler-qwen3 --model-template-token 64 \
  --chat-template-kwargs '{"enable_thinking":false}'
python -c 'import pathlib; p=pathlib.Path("assets/ruler-qwen3"); (p/"all.jsonl").write_text("".join(f.read_text() for f in sorted(p.glob("*-*.jsonl"))))'
```

根据实际 Python、权重、数据、评分源码和设备生成项目私有配置。`fullnet configure` 会写入 `models/<model>/configs/<platform>.local.yaml`，同时记录模型解释器版本和 CANN 路径：

```bash
fullnet configure --model qwen3-0.6b --platform a3 \
  --python "$(command -v python)" --weights assets/weights/qwen3-0.6b \
  --dataset assets/ruler-qwen3/all.jsonl \
  --official-metrics cache/RULER/scripts/eval/synthetic/constants.py --devices 0
```

配置生成后先执行 `fullnet check --config ...`，再 `fullnet plan --config ... --run <id>`。不要直接把带占位符的 `*.example.yaml` 当作真实运行配置；模型运行时、设备号、权重和数据都必须填写实际值。

## 4. 运行与验收

```bash
fullnet check --config models/qwen3-0.6b/configs/a3.local.yaml
fullnet plan --config models/qwen3-0.6b/configs/a3.local.yaml --run a3-native-001
fullnet run --model qwen3-0.6b --run a3-native-001
fullnet report --model qwen3-0.6b --run a3-native-001
```

AI 按 `fullnet-validate` 执行原始配置核对、真实短生成、固定评测、服务性能和设备 profile。HTTP fixture 可以先验证编排和状态记录，但不包含模型推理，不能计入 A3/A5 设备验收。长任务使用 [长任务与 AI 执行](background-jobs.md)，失败定位使用 [调试方法](debugging.md)。
