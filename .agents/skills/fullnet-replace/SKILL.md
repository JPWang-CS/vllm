---
name: fullnet-replace
description: 将独立算子仓构建的扩展或算子包接入指定模型，验证实际调用后与原生基线比较。
---

# 算子替换

读取目标模型 `changes/<change>/change.yaml` 和 `docs/operator-replacement.md`。先明确数学计算、输入/输出、dtype、shape、stride、stream、并行配置和适用范围。

外部包由其自身构建流程产出。可用 Python 扩展、自定义 OPP、ACLNN 动态库或独立可执行程序；框架不依赖特定算子仓。安装与运行目录均使用当前项目的私有环境。

每个变更声明支持平台、依赖、冲突、加载环境和 `verify_argv`。验证程序应执行目标算子，保存实际加载路径、命中分支、输入形状和输出对照。import 成功只证明包可导入。模型适配器还要证明前向确实使用了替换实现。

原生与替换使用独立实验。固定权重、精度、并行布局、数据、模板、生成参数与设备。执行 `fullnet compare` 比较冻结输入与输出文本；逐层/算子导出张量用 `fullnet parity --reference ... --candidate ... --atol ... --rtol ...`。阈值沿用该模型或算子的原验收协议。

确认替换命中和精度后，用 `fullnet-profile` 采集性能。越出优化范围时按算子包的显式回退策略执行，并分别统计命中与回退。
