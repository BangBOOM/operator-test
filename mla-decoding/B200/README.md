# MLA Decoding on B200

This directory records operator benchmark methods for the DeepSeek V3 MLA decoding scenario on NVIDIA B200. On B200, the TRT-LLM MLA kernel is integrated in the FlashInfer repository. This benchmark uses `BatchMLAPagedAttentionWrapper` with the `trtllm-native` backend.

## B200 Environment

Run the following commands on the host. Start the container from the root of this repository:

```bash
IMAGE=nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.8.1.post1
NAME=weyang_dev

docker run -it --name ${NAME} \
  --ipc=host \
  --gpus=all \
  --network host \
  --privileged \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ${PWD}:/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace \
  ${IMAGE}
```

Run all commands below inside the container. After entering the container, set up FlashInfer in the B200 hardware directory:

```bash
cd /workspace/mla-decoding/B200
./setup.sh
```

This command clones or updates:

```text
mla-decoding/B200/flashinfer
```

The setup step also overwrites FlashInfer's benchmark routine with this directory's `attention.py`:

```text
mla-decoding/B200/attention.py
  -> mla-decoding/B200/flashinfer/benchmarks/routines/attention.py
```

The run scripts in this directory execute from this local FlashInfer checkout by default.

## Nsight Systems

To collect profile results with `nsys profile`, install the Nsight Systems CLI inside the container:

```bash
wget -O NsightSystems-linux-cli-public-2026.2.1.210-3763964.deb \
  https://developer.nvidia.com/w/devtools/nsight-systems/NsightSystems-linux-cli-public-2026.2.1.210-3763964.deb

dpkg -i --ignore-depends=libglib2.0-0 \
  ./NsightSystems-linux-cli-public-2026.2.1.210-3763964.deb
```

## B200 FP8 Decoding Single Case

Run the following commands inside the container. Run the configured FP8 decoding case:

```bash
cd /workspace/mla-decoding/B200
./run_mla_single.sh
```

Profile the same case with Nsight Systems:

```bash
cd /workspace/mla-decoding/B200
PROFILE=1 ./run_mla_single.sh
```

Default parameters:

```text
batch_size=16
page_size=32
s_qo=4
s_kv=8192
num_qo_heads=16
num_kv_heads=1
head_dim_ckv=512
head_dim_kpe=64
q_dtype=fp8_e4m3
kv_dtype=fp8_e4m3
num_iters=100
dry_run_iters=10
backend=trtllm-native
```

Raw benchmark output is printed to stdout. When `PROFILE=1`, the default nsys output prefix is:

```text
mla-decoding/B200/results/trtllm_mla_decode
```

Parameters can be overridden with environment variables without editing the script:

```bash
BATCH_SIZE=32 S_QO=1 NUM_QO_HEADS=128 ./run_mla_single.sh
```

## B200 FP8 Decoding Batch Cases

Run the following commands inside the container. Benchmark multiple batch sizes:

```bash
cd /workspace/mla-decoding/B200
./run_mla_bench.sh
```

Default batch sizes:

```text
1 2 4 8 12 16 24 32 48 64 96 128 192 256 512
```

The batch script's default parameters are configured for the DeepSeek V3 decoding shape:

```text
page_size=32
s_qo=1
s_kv=8192
num_qo_heads=128
num_kv_heads=1
head_dim_ckv=512
head_dim_kpe=64
q_dtype=fp8_e4m3
kv_dtype=fp8_e4m3
num_iters=100
dry_run_iters=10
backend=trtllm-native
```

Override batch sizes or other parameters:

```bash
BATCH_SIZES="1 8 16 32" S_QO=4 NUM_QO_HEADS=16 ./run_mla_bench.sh
```

Script output looks like this:

```text
batch_size=1 [PERF] trtllm-native  :: median time 0.016 ms; std 0.000 ms; achieved tflops 140.085 TFLOPs/sec; achieved tb_per_sec 0.298 TB/sec
batch_size=2 [PERF] trtllm-native  :: median time 0.018 ms; std 0.001 ms; achieved tflops 247.158 TFLOPs/sec; achieved tb_per_sec 0.526 TB/sec
batch_size=4 [PERF] trtllm-native  :: median time 0.019 ms; std 0.000 ms; achieved tflops 479.349 TFLOPs/sec; achieved tb_per_sec 1.021 TB/sec
```

If a case fails, the script prints the failed batch size and the full benchmark output to stderr, then exits with a non-zero status.

## Equivalent Raw Command

Run the following command inside the container. The single-case profile is equivalent to running this from the `mla-decoding/B200/flashinfer` directory:

```bash
nsys profile -t cuda,nvtx,osrt,cudnn,cublas \
  -o ../results/trtllm_mla_decode \
  python benchmarks/flashinfer_benchmark.py \
    --routine BatchMLAPagedAttentionWrapper \
    --backends trtllm-native \
    --page_size 32 \
    --batch_size 16 \
    --s_qo 4 \
    --s_kv 8192 \
    --num_qo_heads 16 \
    --num_kv_heads 1 \
    --head_dim_ckv 512 \
    --head_dim_kpe 64 \
    --q_dtype fp8_e4m3 \
    --kv_dtype fp8_e4m3 \
    --num_iters 100 \
    --dry_run_iters 10 \
    -vv
```


| Qlen | KVLen | Qheads | KVHeads | Batch | Duration-MI355(us) | Duration-B200(us) | tflops-mi355 | tflops-b200 |
|------|-------|--------|---------|-------|--------------------|-------------------|--------------|-------------|
| 1 | 8192 | 128 | 1 | 1 | 29.5711 | 16 | 77.15984106 | 142.606336 |
| 1 | 8192 | 128 | 1 | 2 | 33.3271 | 18 | 136.927688 | 253.5223751 |
| 1 | 8192 | 128 | 1 | 4 | 32.8712 | 19 | 277.653554 | 480.3581844 |
| 1 | 8192 | 128 | 1 | 8 | 38.4713 | 22 | 474.4734648 | 829.7095913 |
| 1 | 8192 | 128 | 1 | 12 | 46.7073 | 27 | 586.2127871 | 1014.0895 |
| 1 | 8192 | 128 | 1 | 16 | 55.2839 | 32 | 660.3590198 | 1140.850688 |
| 1 | 8192 | 128 | 1 | 24 | 70.3235 | 42 | 778.6989132 | 1303.829358 |
| 1 | 8192 | 128 | 1 | 32 | 85.8915 | 51 | 850.0776448 | 1431.655765 |
| 1 | 8192 | 128 | 1 | 48 | 117.2 | 70 | 934.4852052 | 1564.595229 |
| 1 | 8192 | 128 | 1 | 64 | 151.6 | 74 | 963.2512405 | 1973.363352 |
| 1 | 8192 | 128 | 1 | 96 | 214.5 | 136 | 1021.181035 | 1610.612736 |
| 1 | 8192 | 128 | 1 | 128 | 282.3 | 140 | 1034.565271 | 2086.126972 |
| 1 | 8192 | 128 | 1 | 192 | 451 | 206 | 971.3673264 | 2126.634292 |
| 1 | 8192 | 128 | 1 | 256 | 565.4 | 283 | 1033.101437 | 2064.012552 |
| 1 | 8192 | 128 | 1 | 512 | 1108.9 | 544 | 1053.504468 | 2147.483648 |
| 2 | 8192 | 128 | 1 | 1 | 42.4794 | 22 | 107.4262525 | 207.4273978 |
| 2 | 8192 | 128 | 1 | 2 | 47.4195 | 20 | 192.4694589 | 456.3402752 |
| 2 | 8192 | 128 | 1 | 4 | 47.835 | 23 | 381.5952965 | 793.6352612 |
| 2 | 8192 | 128 | 1 | 8 | 55.5591 | 29 | 657.0880741 | 1258.869725 |
| 2 | 8192 | 128 | 1 | 12 | 69.4114 | 38 | 788.9314007 | 1441.074553 |
| 2 | 8192 | 128 | 1 | 16 | 87.9396 | 46 | 830.2794649 | 1587.270522 |
| 2 | 8192 | 128 | 1 | 24 | 119.2 | 65 | 918.8059232 | 1684.948708 |
| 2 | 8192 | 128 | 1 | 32 | 156.3 | 67 | 934.2859121 | 2179.535643 |
| 2 | 8192 | 128 | 1 | 48 | 218.6 | 126 | 1002.028052 | 1738.439144 |
| 2 | 8192 | 128 | 1 | 64 | 283.7 | 128 | 1029.459909 | 2281.701376 |
| 2 | 8192 | 128 | 1 | 96 | 439.9 | 191 | 995.8778454 | 2293.647457 |
| 2 | 8192 | 128 | 1 | 128 | 554 | 257 | 1054.360203 | 2272.823161 |
| 2 | 8192 | 128 | 1 | 192 | 893 | 395 | 981.1571426 | 2218.160325 |
| 2 | 8192 | 128 | 1 | 256 | 1129.3 | 472 | 1034.47366 | 2475.065899 |
| 2 | 8192 | 128 | 1 | 512 | 2156.5 | 980 | 1083.451059 | 2384.145111 |
| 4 | 8192 | 128 | 1 | 1 | 21.1711 | 20 | 431.0973688 | 456.3402752 |
| 4 | 8192 | 128 | 1 | 2 | 25.0551 | 22 | 728.5387409 | 829.7095913 |
| 4 | 8192 | 128 | 1 | 4 | 32.2674 | 28 | 1131.396456 | 1303.829358 |
| 4 | 8192 | 128 | 1 | 8 | 48.896 | 43 | 1493.260063 | 1698.010326 |
| 4 | 8192 | 128 | 1 | 12 | 70.2599 | 64 | 1558.807599 | 1711.276032 |
| 4 | 8192 | 128 | 1 | 16 | 81.816 | 64 | 1784.845117 | 2281.701376 |
| 4 | 8192 | 128 | 1 | 24 | 129.4 | 122 | 1692.761454 | 1795.437148 |
| 4 | 8192 | 128 | 1 | 32 | 205 | 125 | 1424.672079 | 2336.462209 |
| 4 | 8192 | 128 | 1 | 48 | 313.6 | 185 | 1396.960026 | 2368.036023 |
| 4 | 8192 | 128 | 1 | 64 | 402.6 | 249 | 1450.858302 | 2345.845591 |
| 4 | 8192 | 128 | 1 | 96 | 609.5 | 380 | 1437.52802 | 2305.719285 |
| 4 | 8192 | 128 | 1 | 128 | 822.2 | 457 | 1420.860015 | 2556.304386 |
| 4 | 8192 | 128 | 1 | 256 | 1814.1 | 936 | 1287.945653 | 2496.220309 |
| 4 | 8192 | 128 | 1 | 512 | 3665.4 | 1894 | 1274.874343 | 2467.225142 |
| 4 | 8192 | 16 | 1 | 1 | 14.6791 | 18 | 77.71938934 | 63.38059378 |
| 4 | 8192 | 16 | 1 | 2 | 16.203 | 16 | 140.8196862 | 142.606336 |
| 4 | 8192 | 16 | 1 | 4 | 17.6392 | 23 | 258.7080339 | 198.4088153 |
| 4 | 8192 | 16 | 1 | 8 | 21.2351 | 31 | 429.7980939 | 294.4130808 |
| 4 | 8192 | 16 | 1 | 12 | 24.6233 | 42 | 555.9859262 | 325.9573394 |
| 4 | 8192 | 16 | 1 | 16 | 28.7992 | 53 | 633.823544 | 344.4077549 |
| 4 | 8192 | 16 | 1 | 24 | 33.7113 | 84 | 812.2029264 | 325.9573394 |
| 4 | 8192 | 16 | 1 | 32 | 40.8832 | 86 | 892.9639073 | 424.5025816 |
| 4 | 8192 | 16 | 1 | 48 | 59.643 | 161 | 918.1435043 | 340.1293977 |
| 4 | 8192 | 16 | 1 | 64 | 69.7393 | 175 | 1046.962674 | 417.2253945 |
| 4 | 8192 | 16 | 1 | 96 | 102.7 | 246 | 1066.423233 | 445.2100246 |
| 4 | 8192 | 16 | 1 | 128 | 128.4 | 317 | 1137.296636 | 460.6589529 |
| 4 | 8192 | 16 | 1 | 256 | 231.6 | 571 | 1261.043938 | 511.4847218 |
| 4 | 8192 | 16 | 1 | 512 | 471.8 | 1095 | 1238.05755 | 533.4388605 |
