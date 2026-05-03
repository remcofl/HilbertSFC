<!-- markdownlint-disable MD033 MD041 -->
---
<h1 align="center">
HilbertSFC
</h1>

<p align="center">
    <strong>Ultra-fast 2D &amp; 3D Hilbert space-filling curve encode/decode kernels for Python.</strong>
</p>

<p align="center">
    <img src="img/hilbert2d_grid.png" width="420" align="middle" alt="2D Hilbert curves for nbits 1..5" />
    <img src="img/hilbert3d_grid.png" width="340" align="middle" hspace="5" alt="3D Hilbert curves animation grid for nbits 1..4" />
</p>

<p align="center">
    <sub>2D Hilbert curves (nbits 1..5) and 3D Hilbert curves (nbits 1..4).</sub>
</p>

<p align="center">
<strong>✨ New in v0.3.0</strong>: PyTorch API + GPU-accelerated kernels with Triton!</br>
<strong>New in v0.4.0</strong>: Morton/z-order curves</br>
</p>

---

This library is performance-first and implemented entirely in Python. It provides fast Hilbert encode/decode kernels for both CPU and GPU, with convenient high-level APIs for NumPy and PyTorch, low-level *kernel accessors*, and clean integration with `torch.compile` for fusion with surrounding operations. For completeness, it also includes Morton/z-order curve kernels.

The hot kernels are JIT-compiled with Numba (CPU) and Triton (GPU) and tuned for:

- Branchless, fully unrolled inner loops
- Small, L1-cache-friendly lookup tables (LUTs)
- Reduced dependency chains for better ILP and MLP (e.g. state-independent lookups)
- Multi-threading for batch processing
- SIMD via LLVM vector intrinsics (CPU)
- Reduced register pressure (GPU)

## When and why to use HilbertSFC?

If you have 2D or 3D coordinates and need a 1D ordering that preserves spatial locality, the Hilbert space-filling curve is a strong choice: points that are close in Euclidean space tend to remain close after mapping to a Hilbert index. HilbertSFC
is designed for high-throughput workloads, such as spatial indexing (GIS/databases), scientific computing, and machine/deep learning, where Hilbert curve mapping performance matters.

## Quick start

Start here: [Quick start](quickstart.md)

## API reference

Browse the full [API reference](api/index.md), or jump directly to:

- [`hilbertsfc`][hilbertsfc]
- [`hilbertsfc.torch`][hilbertsfc.torch]
