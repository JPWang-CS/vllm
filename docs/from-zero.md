# 从零准备环境与模型

初次使用依次准备推理环境、Fullnet Lab 框架、模型、数据和项目私有配置。项目目录保存下载、缓存、实验和日志；CANN、Python、权重和设备按实际环境核对，A2、A3 与 A5 分开验收。

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
  --official-metrics cache/RULER/scripts/eval/synthetic/constants.py --devices 0 \
  --context-limit 16384
```

配置生成后先执行 `fullnet check --config ...`，再 `fullnet plan --config ... --run <id>`。不要直接把带占位符的 `*.example.yaml` 当作真实运行配置；模型运行时、设备号、权重和数据都必须填写实际值。

### 3.1 为公开测量入口准备冻结 contract

`fullnet configure` 只生成基础启动和评测配置，不会猜测性能负载或 profiler 目录。需要 profile/performance 时，在执行 `plan` 之前把下面的合法 mapping 合并到本机配置；这是一个可直接被当前 CLI 和 collector 消费的最小盘古代表 contract：

```yaml
env:
  VLLM_TORCH_PROFILER_DIR: ${RUN}/profiles
  VLLM_TORCH_PROFILER_RECORD_SHAPES: "1"
  VLLM_TORCH_PROFILER_WITH_STACK: "0"
  VLLM_TORCH_PROFILER_WITH_FLOPS: "0"
  VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY: "0"
measurements:
  requests: 32
  warmups: 2
  timeout_per_group: 21600
  deferred_stages: []
  workloads:
    - {input: 128, output: 128, concurrency: 1}
    - {input: 4096, output: 128, concurrency: 1}
    - {input: 15999, output: 128, concurrency: 1}
    - {input: 128, output: 1024, concurrency: 1}
    - {input: 4096, output: 1024, concurrency: 4}
    - {input: 128, output: 4096, concurrency: 4}
```

这 6 组 workload 中最大的 `input + output` 是 `15999 + 128 = 16127`，所以 service argv 的 `--max-model-len` 必须至少为 16127；本示例显式使用 16384。更换上下文上限或 workload 时，逐组重新核对 `input + output <= max_model_len`，不能只看输入长度。

`measurements` 必须是 mapping；`workloads` 必须是非空 list，每项的 `input`、`output`、`concurrency` 都是正整数，`requests` 是正整数，`warmups` 可以为 0 但不能为负数。要做基础运行/仅固定评测，可以把 `profile`、`performance` 写入 `deferred_stages`，但这只表示两个阶段延后，不表示质量通过；此时不能再用对应的 `--profile` 或 `--performance` flag。若要真正采集，应清空 `deferred_stages` 并重新 `plan`。

推荐使用 `VLLM_TORCH_PROFILER_DIR` 启用服务端 profiler。当前 CLI 也会识别 service argv 中带非空值的 `--profiler-config <value>` 或 `--profiler-config=<value>`，但它只检查 argv 中是否出现该选项，不解析配置文件，也不证明目标 vLLM service 支持该选项或文件格式。使用这种方式前，必须用目标 service 的实际 `--help` 或源码确认选项语义；如果目标接口要求 JSON 字符串，应按目标接口的格式传值，不能把 CLI 的存在性检查当作 profiler 已启用。

准备好 contract 后重新冻结并运行：

```bash
fullnet check --config models/qwen3-0.6b/configs/a3.local.yaml
fullnet plan --config models/qwen3-0.6b/configs/a3.local.yaml --run a3-native-profile-001
fullnet run --model qwen3-0.6b --run a3-native-profile-001 --profile --performance
```

`--pd-probe mixed` 只用于已经具备同实例混合 P/D 运行证据的服务；`--pd-probe split` 需要实际的 producer/consumer、KV-transfer connector、split front 和 `/metrics_snapshot`，不能对 `fullnet configure` 生成的普通单服务 argv 直接套用。P/D 拓扑、角色、设备分组或服务 argv 改变时，必须建立新的配置和 run。

### 3.2 设备选择与 A2 基础功能

配置里的 `devices` 是传给运行时的 logical/runtime selector。它不是 `npu-smi` 输出中的物理 board，也不能由 Slot ID、Chip Phy-ID 或表格行号推断 runtime。A2、A3、A5 启动前都要在目标 host 上保存当前 ACL 物理到 user/runtime 映射，核对目标物理卡健康、空闲、无本任务外进程，并在同一次 fresh gate 后启动；换卡、换 host 或换平台都要重新做这一步。

A2 的基础功能使用同一条 `configure → check → plan → run` 路径，把 `--platform a2` 和实际 A2 runtime selector 写入新配置。基础功能包括真实服务启动、短请求、冻结评测和 cleanup；没有质量阈值时只记录 `quality_status=not_specified`，不能把 A2 的协议完成写成质量通过，也不能用 A3/A5 的设备或性能结果代替 A2。

## 4. 运行与验收

```bash
fullnet check --config models/qwen3-0.6b/configs/a3.local.yaml
fullnet plan --config models/qwen3-0.6b/configs/a3.local.yaml --run a3-native-001
fullnet run --model qwen3-0.6b --run a3-native-001
fullnet report --model qwen3-0.6b --run a3-native-001
```

AI 按 `fullnet-validate` 执行原始配置核对、真实短生成、固定评测、服务性能和设备 profile。HTTP fixture 可以先验证编排和状态记录，但不包含模型推理，不能计入 A2/A3/A5 设备验收。`fullnet run` 返回 0 表示框架任务和评测协议完成；它不把 `quality_status=not_specified` 改成 `passed`。交付脚本还要检查结果 JSON 中的 `quality_status`、profile、performance 和 cleanup。长任务使用 [长任务与 AI 执行](background-jobs.md)，失败定位使用 [调试方法](debugging.md)。
