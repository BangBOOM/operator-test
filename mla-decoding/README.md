# MLA Decoding

This directory records operator benchmark methods for the DeepSeek V3 MLA decoding scenario. MLA decoding is an attention-related operator comparison item in the decode stage of DeepSeek V3 inference. The current focus is latency across different hardware and backends.

This README only describes the operator itself and the directory organization. Details about starting the hardware environment, setting up dependent repositories, and running test scripts are documented in the README for each hardware directory.

## Hardware Directories

| Directory | Hardware | Description |
| --- | --- | --- |
| `B200/` | NVIDIA B200 | TensorRT-LLM native backend in FlashInfer |
| `MI355/` | AMD MI355 | `test_mla_persistent.py` under `/sgl-workspace/aiter` in the SGLang ROCm container |
