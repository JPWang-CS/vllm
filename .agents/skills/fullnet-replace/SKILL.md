---
name: fullnet-replace
description: 将独立算子仓构建的扩展或算子包接入指定模型，验证实际调用后与原生基线比较。
---

# 算子替换

读取目标模型的变更配置和 `docs/operator-replacement.md`，先明确数学计算、输入/输出、dtype、shape、stride、stream、并行和适用范围。

1. 外部包由算子仓构建。运行目录、加载路径和缓存使用当前项目的私有环境；记录实际 import 文件、动态库、OPP 路径和运行时版本。import 成功只证明包可导入。
2. 每个变更声明支持平台、依赖、冲突、加载环境和 `verify_argv`。验证程序必须证明目标算子实际执行、命中分支、输入形状和输出对照；模型适配器还要证明前向确实使用替换实现。
3. 原生与替换使用独立实验，固定权重、精度、并行布局、数据、模板、生成参数和设备。文本结果用 `fullnet compare`；张量结果用 `fullnet parity --reference ... --candidate ... --atol ... --rtol ...`，阈值沿用原验收协议。
4. 先确认替换命中和精度，再用 `fullnet-profile` 采集性能。profile daemon 失败时保留 task 副本并恢复离线索引；CPU trace 不能作为设备收益证据。
5. 越出优化范围时按显式回退策略执行，分别统计命中与回退。换卡、换并行或改变负载后重新建立原生/替换成对结果。
