# Fullnet Lab

整网推理实验框架，用于运行模型、验证精度、分析性能，并比较算子替换对整网的影响。支持按 A3、A5 分别维护环境和实验；模型项目与公共工具保存在同一仓库。

## 安装

Python 3.10 或更新版本。安装器在项目内创建独立环境；系统缺少 pip 时可用官方 virtualenv 引导：

```bash
python scripts/bootstrap.py --extras test
source envs/fullnet/bin/activate
```

框架编排层仅依赖 PyYAML。模型使用其已验证的 PyTorch、torch_npu、vLLM 和平台插件环境，框架安装不升级这些软件。CANN 在进入任务时手动 source。

## 从一个模型开始

```text
models/
  pangu-7b/
    model.yaml
    configs/          A3/A5 配置与本机配置
    changes/          算子替换、图模式等独立变更
    baselines/        已接受实验的索引
    experiments/      每次运行的配置、日志、结果和报告
  pangu-92b/
src/fullnet/          公共运行、评测、比较与报告
.agents/skills/       AI 工作流程
scripts/             数据准备与受控上机入口
docs/                评测、替换和性能方法
```

复制目标平台配置为本机 `.local.yaml`，填写设备和环境变量。先验证和保存配置，再运行：

```bash
fullnet check --config models/pangu-7b/configs/a3.local.yaml
fullnet plan --config models/pangu-7b/configs/a3.local.yaml --run a3-native-001
# 在项目个人 tmux 会话内执行，任务命令和日志保存在项目目录
fullnet run --model pangu-7b --run a3-native-001
fullnet report --model pangu-7b --run a3-native-001
```

`plan` 保存实际配置和固定输入。`run` 启动本次服务、检查模型名、发出真实请求、评分并结束本次进程组。任务得分、数值一致性和资源释放分别记录，失败样本保留在评测分母中。

AI 可按自然语言运行：“在 A3 上启动 Pangu 7B，沿用已有固定评测数据，保存原生基线。”项目 [AGENTS.md](AGENTS.md) 路由到相应 skill。服务器执行遵循个人容器、私有环境和项目目录边界。

## 本地检查工作流程

以下示例启动一个回显 HTTP fixture，检查配置、状态、请求、评分和收尾；无需模型和设备：

```bash
export FULLNET_PYTHON="$(command -v python)"
fullnet plan --config examples/offline/config.yaml --run workflow-001
fullnet run --model workflow-fixture --run workflow-001
fullnet report --model workflow-fixture --run workflow-001
python -m pytest -q
```

## 算子替换与性能

外部算子包通过模型目录中的变更配置与适配器接入。原生、替换分别运行，同条件比较。编译、打包由算子仓维护，框架接收其独立产物。

- [运行与评测](docs/evaluation.md)
- [长任务与 AI 执行](docs/background-jobs.md)
- [从零准备环境与模型](docs/from-zero.md)
- [外部算子接入](docs/operator-replacement.md)
- [性能分析](docs/profiling.md)
- [实现与验证记录](STATUS.md)
- [Pangu 7B](models/pangu-7b/README.md) · [Pangu 92B](models/pangu-92b/README.md)

服务性能与设备 profiler 的正式采集随模型实验配置执行。当前实现包含启动、固定样本评测、原生/替换输出比较和张量数值检查；实际 A3/A5 验证状态见各模型项目记录。
