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
    <sub>2D Hilbert curves (nbits 1..5) and 3D Hilbert curves (nbits 1..4, animated).</sub>
</p>

---

This project is **performance-first** and **implemented entirely in Python**. The hot kernels are JIT-compiled with Numba and tuned for:

- **Branchless, fully unrolled inner loops**
- **SIMD via LLVM vector intrinsics**
- **Small, L1-cache-friendly lookup tables (LUTs)**
- **Reduced dependency chains for better ILP, e.g., state-independent gather**
- **Optional multi-threading for batch operations**

It provides both convenient Python APIs and *kernel accessors* designed to be embedded into other Numba kernels.

## Quick start

Start here: [Quick start](quickstart.md)

## API reference

- [`hilbert_encode_2d`][hilbertsfc.hilbert2d.hilbert_encode_2d]
- [`hilbert_decode_2d`][hilbertsfc.hilbert2d.hilbert_decode_2d]
- [`hilbert_encode_3d`][hilbertsfc.hilbert3d.hilbert_encode_3d]
- [`hilbert_decode_3d`][hilbertsfc.hilbert3d.hilbert_decode_3d]

For kernel accessors, cache helpers, and typing aliases, use the navigation bar.
