# Operator Test

这个仓库用于沉淀不同算子的性能对比方式。每个算子使用一个独立目录，目录内需要说明和提供：

- 测试环境如何启动
- 依赖的上游仓库如何准备
- 单 case 如何运行
- 批量 case 如何运行
- 耗时结果如何获取

## 算子目录

| 目录 | 算子 | 当前目标 |
| --- | --- | --- |
| `mla-decoding/` | MLA paged attention decoding | DeepSeek V3 MLA；按硬件平台分目录，目前包含 B200、MI355 |

## Agent Setup 约定

当用户让 agent `setup` 某个 kernel/operator 时，进入对应算子和硬件目录执行该目录下的 setup 脚本。例如：

```bash
cd mla-decoding/B200
./setup.sh
```

setup 脚本会把上游 benchmark 仓库 clone 或更新到硬件目录内部。这样后续运行脚本时不需要手动切路径，直接在硬件目录调用即可。
