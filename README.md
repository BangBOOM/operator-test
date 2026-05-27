# Operator Test

This repository records performance comparison methods for different operators. Each operator uses a separate directory, and each directory should document and provide:

- How to start the test environment
- How to prepare dependent upstream repositories
- How to run a single case
- How to run batch cases
- How to collect latency results

## Operator Directories

| Directory | Operator | Current Target |
| --- | --- | --- |
| `mla-decoding/` | MLA paged attention decoding | DeepSeek V3 MLA; organized by hardware platform, currently including B200 and MI355 |

## Agent Setup Convention

When the user asks the agent to `setup` a kernel/operator, enter the corresponding operator and hardware directory and run the setup script in that directory. For example:

```bash
cd mla-decoding/B200
./setup.sh
```

The setup script clones or updates the upstream benchmark repository inside the hardware directory. After that, scripts can be run directly from the hardware directory without manually changing paths.
