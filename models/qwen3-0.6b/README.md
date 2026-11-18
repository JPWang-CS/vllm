# Qwen3 0.6B

公开的小模型入口，用于从零检查下载、tokenizer、服务启动与评测流程。下载来源为 [Qwen 官方模型](https://huggingface.co/Qwen/Qwen3-0.6B)。

```bash
python -m pip install -e '.[download,evaluation]'
fullnet assets plan --model qwen3-0.6b
fullnet assets fetch --model qwen3-0.6b
fullnet assets check --model qwen3-0.6b
```

模型文件保存到 `assets/weights/qwen3-0.6b/`。访问凭据从 `HF_TOKEN` 读取，保留在用户环境中。下载失败保留 Hub 断点；同一命令可以继续下载。文件齐全只说明数据已获取，模型执行由整网实验另行验证。

A3/A5 使用对应的 vLLM Ascend 环境，步骤见 `docs/from-zero.md`。模型与 Pangu 的实验分开保存，得分只与本模型对应基线比较。
