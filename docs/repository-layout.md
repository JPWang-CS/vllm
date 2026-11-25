# Fullnet Lab 模型与实验目录规范

本规范统一模型、配置、算子变更、基线和实验的仓内归属，保证部署、精度和性能结果可以从模型一路追溯到配置、run 和报告。

## 目录结构

```text
models/<model>/
├── model.yaml                 # 模型身份、来源、平台和运行声明
├── README.md                  # 模型专用说明、权重来源和运行约束
├── configs/                   # 示例配置与 fullnet configure 生成的本地配置
│   ├── <platform>.example.yaml
│   └── <platform>.local.yaml
├── adapters/                  # 模型专用服务或并行适配，可选
├── changes/                   # 算子替换或其他受控变更，可选
│   └── <change-id>/change.yaml
├── baselines/                 # 已接受基线索引，可选
│   └── index.yaml
└── experiments/               # fullnet plan 创建的模型级实验
    └── <run-id>/
```

`models/<model>/` 是配置、适配器、变更、基线和实验的一级归属。`src/fullnet/` 保存通用框架代码；`assets/weights/<model>/` 保存按模型区分的权重；`cache/`、`envs/`、`runs/` 和 `tmp/` 保存项目级缓存、环境或任务辅助数据。模型运行结果仍须回到对应的 `models/<model>/experiments/<run-id>/`。

模型声明中的 `name` 与模型目录名一致。平台、权重、dtype、量化、并行度、运行时版本和设备写入该模型的实际配置，不跨模型复用本地配置。

## 实验阶段和归属

原生 baseline 使用一个实验，替换 candidate 使用另一个实验。每个实验都可以按合同依次完成精度、性能和 profile；只有输入、配置或实现合同改变时才创建新 run。

| 阶段 | 归属和内容 |
|---|---|
| baseline | `models/<model>/baselines/index.yaml` 引用本模型已接受的 `experiments/<run-id>/`。 |
| replacement | `models/<model>/changes/<change-id>/change.yaml` 描述变更；实验配置和生成的 `changes.json` 记录启用项。 |
| precision | 在对应的 baseline 或 candidate 实验内保存输入、dtype、版本、输出和误差合同。 |
| profile | 在对应实验的精度合同通过后保存性能负载、采样参数、设备证据和报告。 |

同一个 run 对应一个明确的输入、配置和实现合同。`src/fullnet/execution.py` 的 `plan()` 创建并冻结 run，`run_experiment()` 在该 run 内记录启动、smoke、evaluation、profile、performance 和 cleanup 等阶段；实验已经开始后不能复用同一 run。精度证据未通过时，profile 保持未放行；合同改变时保留旧 run 并创建新 run。

## 新增模型

新增模型建立 `model.yaml` 和 `README.md`，按需要增加 `configs/`、`adapters/`、`changes/`、`baselines/` 和 `experiments/`。不需要的可选目录不提前创建。

模型下载和配置使用现有 CLI：

```bash
fullnet assets plan --model <model>
fullnet assets verify-source --model <model>
fullnet assets fetch --model <model>
fullnet assets check --model <model>
fullnet configure --model <model> --platform <platform> \
  --python <model-python> --weights <weights> --dataset <dataset> \
  --official-metrics <metrics> --devices <device-id>
fullnet check --config models/<model>/configs/<platform>.local.yaml
fullnet plan --config models/<model>/configs/<platform>.local.yaml --run <run-id>
```

`fullnet configure` 写入模型自己的 `configs/`；`fullnet plan` 创建模型自己的 `experiments/<run-id>/`。随后按实验合同执行 `fullnet run`、`fullnet status`、`fullnet report`、`fullnet compare`、`fullnet parity` 或 `fullnet export`。

## 现有模型的新部署和算子变更

现有模型的新平台、新服务参数或新对照继续使用原模型目录：

- 配置写入 `models/<model>/configs/`；
- 模型专用服务或并行逻辑写入 `models/<model>/adapters/`；
- 算子替换定义写入 `models/<model>/changes/<change-id>/change.yaml`；
- 原生 baseline 和替换 candidate 使用不同 run ID；各自 run 内按合同记录精度、性能和 profile；
- 每个结果都能从 `model → config → run → report` 追溯。

通用编排继续使用 `src/fullnet/` 和仓库已有脚本，不为单个部署复制一套框架目录。

## 外部 wheel、OPP 和动态库

外部 wheel、OPP 包和 `.so` 保持在其授权的运行环境或项目私有安装位置。模型目录记录引用、版本和验证方式，不把环境的 `site-packages` 或共享工具链复制到 `src/fullnet/`。

运行时路径通过配置的 `env` 或 change 的 `env` 表达；`verify_argv` 验证实际加载路径、符号或 dispatch 证据。外部产物的版本、平台、dtype、shape 和回退条件写入 change 或实验配置。CUDA、ROCm、Ascend 及其他后端共用上述模型级目录合同。

## 验收记录

一个实验目录至少保留冻结后的 `config.json`、`dataset.jsonl`、`state.json`、命令记录、原始输出、日志和报告；性能实验另保留 `performance/` 下的采样参数及 profile 产物。失败、跳过、超时和通过分别记录，原始证据不覆盖。
