# MLA Decoding on MI355

This directory records operator benchmark methods for the DeepSeek V3 MLA decoding scenario on AMD MI355. MI355 tests use the `rocm/sgl-dev:sglang-0.5.10-rocm720-mi35x-mori-0513` image, and the aiter environment path inside the container is `/sgl-workspace/aiter`.

## MI355 Environment

Run the following commands on the host to start the MI355 test container:

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

Enter the container:

```bash
sudo docker exec -it "$NAME" bash
```

Run all commands below inside the container. After entering the container, confirm that `/sgl-workspace/aiter` exists. Put the scripts in this directory and `test_mla_persistent_prof.py` inside the container, or enter this directory through the actual mounted path and run:

```bash
cd <operator-test>/mla-decoding/MI355
./setup.sh
```

`setup.sh` checks:

- `/sgl-workspace/aiter`
- `test_mla_persistent_prof.py` in this directory
- The torch profile output directory, defaulting to `/home/weyang/log/mla_traces`

When running the benchmark, the script directly executes this directory's `test_mla_persistent_prof.py`; there is no need to `cd /sgl-workspace/aiter`. Python and aiter dependencies use the environment already configured inside the container.

## Batch Tests

Run the following commands inside the container. By default, this runs an FP8 decoding batch-size sweep for TP1 and Qlen1:

```bash
cd <operator-test>/mla-decoding/MI355
./run_mla_bench.sh
```

Default batch sizes:

```text
1 2 4 8 12 16 24 32 48 64 96 128 192 256 512
```

Default parameters:

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

The `-n` parameter is generated automatically from `TOTAL_QO_HEADS / TP,QLEN`:

```bash
TP=1 QLEN=1 ./run_mla_bench.sh   # -n 128,1
TP=8 QLEN=4 ./run_mla_bench.sh   # -n 16,4
```

You can also override `-n` directly:

```bash
N_ARG=16,4 ./run_mla_bench.sh
```

Run multiple TP/Qlen configurations in one pass:

```bash
CONFIGS="1:1 8:4" ./run_mla_bench.sh
```

Override batch sizes:

```bash
BATCH_SIZES="1 8 16 32" TP=8 QLEN=4 ./run_mla_bench.sh
```

If the Python command in the container is `python3`:

```bash
PYTHON_BIN=python3 ./run_mla_bench.sh
```

Each case first prints a case header, then keeps only latency lines matching `mla_a8w8_.*`:

```text
=== TP: 1 QLEN: 1 N_ARG: 128,1 Batch Size: 1 ===
mla_a8w8_...
```

If the benchmark command fails, the script prints the corresponding TP, Qlen, batch size, and full output, then exits with a non-zero status.

## Equivalent Raw Command

Run the following command inside the container. TP1, Qlen1, and batch size 1 are equivalent to running this from the `mla-decoding/MI355` directory:

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
