# MLA Decoding on MI355

这个目录记录 AMD MI355 上 DeepSeek V3 MLA decoding 场景的算子 benchmark 方法。MI355 测试使用 `rocm/sgl-dev:sglang-0.5.10-rocm720-mi35x-mori-0513` 镜像，aiter 环境路径在容器内的 `/sgl-workspace/aiter`。

## MI355 环境

以下命令在宿主机执行，用于启动 MI355 测试容器：

```bash
IMG=rocm/sgl-dev:sglang-0.5.10-rocm720-mi35x-mori-0513
NAME=weyang_mi355_dev

sudo docker run -d \
  --privileged \
  -w /root \
  --name "$NAME" \
  --shm-size=128g \
  --device=/dev/infiniband/rdma_cm \
  --device=/dev/infiniband/uverbs0 \
  --device=/dev/infiniband/uverbs1 \
  --device=/dev/infiniband/uverbs2 \
  --device=/dev/infiniband/uverbs3 \
  --device=/dev/infiniband/uverbs4 \
  --device=/dev/infiniband/uverbs5 \
  --device=/dev/infiniband/uverbs6 \
  --device=/dev/infiniband/uverbs7 \
  --ulimit memlock=-1 \
  --ulimit nofile=65536:65536 \
  --network=host \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -v /apps/:/apps/ \
  -v /usr/sbin/nicctl:/usr/sbin/nicctl \
  -v /nfsdata:/nfsdata \
  "$IMG" \
  bash -c "while true; do sleep 3600; done"
```

进入容器：

```bash
sudo docker exec -it "$NAME" bash
```

下面的命令都在容器内执行。进入容器后，确认 `/sgl-workspace/aiter` 已存在。把本目录脚本和 `test_mla_persistent_prof.py` 放到容器内，或者按实际挂载路径进入本目录后执行：

```bash
cd <operator-test>/mla-decoding/MI355
./setup.sh
```

`setup.sh` 会检查：

- `/sgl-workspace/aiter`
- 本目录下的 `test_mla_persistent_prof.py`
- torch profile 输出目录，默认 `/home/weyang/log/mla_traces`

运行 benchmark 时，脚本会直接执行本目录的 `test_mla_persistent_prof.py`，不需要 `cd /sgl-workspace/aiter`。Python 和 aiter 依赖使用容器内已经配置好的环境。

## 批量测试

以下命令在容器内执行。默认运行 TP1、Qlen1 的 FP8 decoding batch size 扫描：

```bash
cd <operator-test>/mla-decoding/MI355
./run_mla_bench.sh
```

默认 batch size：

```text
1 2 4 8 12 16 24 32 48 64 96 128 192 256 512
```

默认参数：

```text
dtype=fp8
kv_dtype=fp8
context_len=8192
total_qo_heads=128
tp=1
qlen=1
-n=128,1
k=512
qn=512
qr=64
vh=512
blk=1
ms=32
torch_profile=/home/weyang/log/mla_traces
torch_profile_warmup=3
torch_profile_iters=10
grep_pattern=mla_a8w8_.*
python_bin=python
```

`-n` 参数由 `TOTAL_QO_HEADS / TP,QLEN` 自动生成：

```bash
TP=1 QLEN=1 ./run_mla_bench.sh   # -n 128,1
TP=8 QLEN=4 ./run_mla_bench.sh   # -n 16,4
```

也可以直接覆盖 `-n`：

```bash
N_ARG=16,4 ./run_mla_bench.sh
```

一次跑多组 TP/Qlen：

```bash
CONFIGS="1:1 8:4" ./run_mla_bench.sh
```

覆盖 batch size：

```bash
BATCH_SIZES="1 8 16 32" TP=8 QLEN=4 ./run_mla_bench.sh
```

如果容器里 Python 命令是 `python3`：

```bash
PYTHON_BIN=python3 ./run_mla_bench.sh
```

每个 case 会先打印 case header，然后只保留匹配 `mla_a8w8_.*` 的耗时行：

```text
=== TP: 1 QLEN: 1 N_ARG: 128,1 Batch Size: 1 ===
mla_a8w8_...
```

如果 benchmark 命令失败，脚本会打印对应 TP、Qlen、batch size 和完整输出，然后退出非 0。

## 等价原始命令

以下命令在容器内执行。TP1、Qlen1、batch size 1 等价于在 `mla-decoding/MI355` 目录执行：

```bash
python test_mla_persistent_prof.py \
  -d fp8 -kvd fp8 \
  -c 8192 \
  -b 1 \
  -n 128,1 \
  -k 512 -qn 512 -qr 64 -vh 512 \
  -blk 1 \
  -ms 32 \
  --torch-profile /home/weyang/log/mla_traces \
  --torch-profile-warmup 3 \
  --torch-profile-iters 10 2>&1 | grep -E "mla_a8w8_.*"
```
