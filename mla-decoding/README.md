# MLA Decoding

这个目录记录 DeepSeek V3 MLA decoding 场景的算子 benchmark 方法。MLA decoding 是 DeepSeek V3 推理 decode 阶段的 attention 相关算子对比项，当前主要关注不同硬件和 backend 下的耗时表现。

本层 README 只描述算子本身和目录组织。具体硬件环境如何启动、依赖仓库如何 setup、测试脚本如何执行，放在对应硬件目录的 README 中。

## 硬件目录

| 目录 | 硬件 | 说明 |
| --- | --- | --- |
| `B200/` | NVIDIA B200 | FlashInfer 中的 TensorRT-LLM native backend |
| `MI355/` | AMD MI355 | SGLang ROCm 容器内 `/sgl-workspace/aiter` 的 `test_mla_persistent.py` |
