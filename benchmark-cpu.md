<!-- markdownlint-disable MD024 -->
# Benchmark (CPU)

This page provides additional benchmark results for Hilbert curve encoding and decoding for 2D and 3D coordinates across multiple Python and Rust implementations, including **HilbertSFC**. The goal of these benchmarks is to compare raw kernel throughput under controlled and reproducible conditions.

## Changelog

- **2026-03-18 (v0.2):** Updated **hilbertsfc** benchmark results.

## Test Methodology

### Experiments

For each implementation, two experiments are executed:

- **Encode (enc)**: map sampled coordinates → Hilbert index
- **Decode (dec)**: map sampled Hilbert indices → coordinates

For each experiment, the following metrics are recorded:

- **Throughput**: average Mpts/s (millions of points per second)
- **Latency**: median ns/pt (nanoseconds per point)

### Benchmark Configuration

Unless otherwise noted, all experiments use:

- Points: `5,000,000`
- Minimum measurement time: `20 s`
- Trials: `1` (minimum number of trials to reach minimum time)
- Mode: `random` (For `ndim = d` and `nbits = b`, coordinates are sampled uniformly from:
$[0, 2^b)^d $)
- Fixed RNG seed (identical input across implementations)
- Single-threaded execution (`--threads 1`)

For each implementation at least one trial is executed, and the trials are repeated until the minimum time is reached. When multiple trials are executed, the median times are reported.

### System

- **CPU**: Intel Core Ultra 7 258v
- **OS**: Ubuntu 24.04.4
- **Python**: 3.12.12
- **Numba**: 0.63.1

## Benchmark Drivers

Benchmarks are executed using two drivers:

- `bench/bench_cli.py`: Primary Python benchmark driver used for all Python implementations
- `bench/hilbert-bench-rs/`: Rust benchmark driver that replicates the Python benchmark, used for Rust implementations

Both drivers:

- Use identical sampling models
- Use identical random seeds
- Measure steady-state throughput over a minimum runtime window

Minor differences in runtime plumbing may exist, but measurements are designed to be directly comparable.

## Results

### 2D Coordinates

These are the results for the implementations on 2D coordinates across different number of coordinate bits (`nbits`). These results show that HilbertSFC is an impressive 3-4 orders of magnitude faster than existing Python implementations, and ~8-11x faster than the fastest Rust implementation.

HilbertSFC can encode around **2 billion** `uint8` points (x, y) per second, and around **500 million** `uint32` points per second, on a single thread. This equates to roughly 8 GB/s of combined read/write bandwidth for 2D encode.

#### `nbits=8`

- Coordinate dtype: `uint8`
- Index dtype: `uint16`

| Implementation | Language | ns/pt (enc) | ns/pt (dec) | Mpts/s (enc) | Mpts/s (dec) |
| --- | --- | ---: | ---: | ---: | ---: |
| 🔥**hilbertsfc** | **Python (+ Numba)** | **0.49** | **0.65** | **2028.64** | **1549.33** |
| fast_hilbert | Rust | 4.46 | 4.43 | 224.36 | 225.52 |
| hilbert_2d | Rust | 29.94 | 25.17 | 33.40 | 39.74 |
| hilbert-bytes | Python (+ Numba) | 601.24 | 498.04 | 1.663 | 2.008 |
| numpy-hilbert-curve | Python (+ NumPy) | 1203.58 | 646.73 | 0.831 | 1.546 |
| hilbertcurve | Python | 4555.00 | 2502.72 | 0.220 | 0.400 |

#### `nbits=16`

- Coordinate dtype: `uint16`
- Index dtype: `uint32`

| Implementation | Language | ns/pt (enc) | ns/pt (dec) | Mpts/s (enc) | Mpts/s (dec) |
| --- | --- | ---: | ---: | ---: | ---: |
| 🔥**hilbertsfc** | **Python (+ Numba)** | **0.78** | **1.00** | **1280.51** | **1002.74** |
| fast_hilbert | Rust | 7.65 | 7.27 | 130.79 | 137.64 |
| hilbert_2d | Rust | 60.93 | 54.20 | 16.41 | 18.45 |
| hilbert-bytes | Python (+ Numba) | 1303.20 | 1091.04 | 0.767 | 0.917 |
| numpy-hilbert-curve | Python (+ NumPy) | 2925.62 | 1639.47 | 0.342 | 0.610 |
| hilbertcurve | Python | 7461.09 | 4700.11 | 0.134 | 0.213 |

#### `nbits=32` (2D maximum)

- Coordinate dtype: `uint32`
- Index dtype: `uint64`

| Implementation | Language | ns/pt (enc) | ns/pt (dec) | Mpts/s (enc) | Mpts/s (dec) |
| --- | --- | ---: | ---: | ---: | ---: |
| 🔥**hilbertsfc (Python)** | **Python (+ Numba)** | **1.38** | **1.59** | **726.68** | **629.52** |
| fast_hilbert | Rust | 12.24 | 12.03 | 81.67 | 83.11 |
| hilbert_2d | Rust | 121.23 | 101.34 | 8.25 | 9.87 |
| hilbert-bytes | Python (+ Numba) | 2997.51 | 2642.86 | 0.334 | 0.378 |
| numpy-hilbert-curve | Python (+ NumPy) | 7606.88 | 5075.58 | 0.131 | 0.197 |
| hilbertcurve | Python | 14355.76 | 10411.20 | 0.0697 | 0.0961 |

### 3D Coordinates

The rust implementations do not support 3D coordinates, so only Python implementations are benchmarked here. Again, we see that HilbertSFC is 3-4 orders of magnitude faster than existing Python implementations, with a throughput around **700 million** `uint8` points (x, y, z) per second, and around **150 million** `uint32` points per second, on a single thread.

For the case of `nbits=21` (the maximum for 3D), we also included results on multi-threading (8 threads), to show that it can further boost throughput by around 4x, reaching around **500 million** `uint32` points per second.

#### `nbits=8`

- Coordinate dtype: `uint8`
- Index dtype: `uint32`

| Implementation | Language | ns/pt (enc) | ns/pt (dec) | Mpts/s (enc) | Mpts/s (dec) |
| --- | --- | ---: | ---: | ---: | ---: |
| 🔥**hilbertsfc** | **Python (+ Numba)** | **1.43** | **1.86** | **697.146** | **537.403** |
| hilbert-bytes | Python (+ Numba) | 913.51 | 755.23 | 1.095 | 1.324 |
| numpy-hilbert-curve | Python (+ NumPy) | 1957.92 | 1013.06 | 0.511 | 0.987 |
| hilbertcurve | Python | 5523.18 | 3048.29 | 0.181 | 0.328 |

#### `nbits=21` (3D maximum)

- Coordinate dtype: `uint32`
- Index dtype: `uint64`

| Implementation | Language | ns/pt (enc) | ns/pt (dec) | Mpts/s (enc) | Mpts/s (dec) |
| --- | --- | ---: | ---: | ---: | ---: |
| 🔥**hilbertsfc (multi-threaded)** | **Python (+ Numba)** | **1.73** | **1.99** | **576.921** | **502.588** |
| 🔥**hilbertsfc** | Python (+ Numba) | 6.64 | 8.23 | 150.548 | 121.501 |
| hilbert-bytes | Python (+ Numba) | 3271.85 | 2799.62 | 0.306 | 0.357 |
| numpy-hilbert-curve | Python (+ NumPy) | 6928.09 | 4049.03 | 0.144 | 0.247 |
| hilbertcurve | Python | 11399.89 | 7926.70 | 0.088 | 0.126 |

### Scalability with Threads

HilbertSFC exhibits near-perfect linear scaling with thread count up to the point of memory bandwidth saturation. Because the kernel is highly optimized, only a small number of cores are required to reach this point.

Due to the hybrid architecture of the Intel CPU (Lion Cove + Skymont cores) used in the previous experiments, this experiment is performed on a uniform-core CPU to demonstrate scalability without the confounding effects of heterogeneous cores.
> **System info:** Amd Ryzen 5700, Dual-channel DDR4 @ 3200 MT/s , Ubuntu 24.04.4, Python 3.12.12, Numba 0.63.1

#### 2D Coordinates, `nbits=32`, `n=20,000,000`

| Threads | ns/pt (enc) | ns/pt (dec) | Mpts/s (enc) | Mpts/s (dec) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.13 | 1.75 | 470.41 | 571.36 |
| (-50%) 2 | 1.06 | 0.87 | 940.07 | 1153.01 |
| (-66%) 3 | 0.72 | 0.66 | 1389.22 | 1513.24 |
| (-71%) 4 | 0.62 | 0.62 | 1614.83 | 1615.04 |
| (-71%) 5 | 0.61 | 0.61 | 1629.04 | 1627.76 |

At 4 threads, encode throughput reaches ~**1.6 Gpts/s**. For 2D encode with `uint32` coordinates and `uint64` indices, the memory traffic per point is:

- Read: 2 × `uint32` = 8 bytes
- Write: 1 × `uint64` = 8 bytes
- Total: 16 bytes / point

This implies an effective combined memory bandwidth of:

$1.6 \times 10^9 \:\mathrm{pts/s} \times 16 \: \mathrm{bytes/pts} \approx 25.6 \:\mathrm{GB/s}$

This is around 80% of sustained STREAM copy bandwidth (not `memcpy`) for this system, indicating the kernel is fundamentally memory-bandwidth bound at higher thread counts. 3D encoding/decoding may benefit from higher thread counts before saturating bandwidth, due to its higher compute intensity.
