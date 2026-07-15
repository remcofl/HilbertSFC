<!-- markdownlint-disable MD033 MD041 -->

<section class="hsfc-hero">
  <h1>HilbertSFC</h1>
  <p class="hsfc-hero__lede">
    Ultra-fast 2D &amp; 3D Hilbert space-filling curve kernels for Python.
  </p>
  <div class="hsfc-hero__actions">
    <a class="hsfc-button hsfc-button--primary" href="quickstart/">Get started <span aria-hidden="true">→</span></a>
    <a class="hsfc-button" href="api/">Explore the API</a>
  </div>
  <div class="hsfc-hero__visuals">
    <img src="img/hilbert2d_grid.png" alt="2D Hilbert curves for nbits 1 through 5" />
    <img src="img/hilbert3d_grid.png" alt="Animated 3D Hilbert curves for nbits 1 through 4" />
  </div>
</section>

## Built for throughput

HilbertSFC is performance-first and implemented entirely in Python. It provides fast Hilbert encode/decode kernels for both CPU and GPU, convenient high-level APIs for NumPy and PyTorch, low-level *kernel accessors*, and clean integration with `torch.compile`. For completeness, it also includes Morton/z-order curve kernels.

The hot kernels are JIT-compiled with Numba on CPU and Triton on GPU and tuned for:

<ul class="hsfc-performance-list">
  <li>Branchless, fully unrolled inner loops</li>
  <li>Small, L1-cache-friendly lookup tables</li>
  <li>Reduced dependency chains for better ILP and MLP</li>
  <li>Multi-threaded batch processing</li>
  <li>SIMD through LLVM vector intrinsics</li>
  <li>Reduced register pressure on GPU</li>
</ul>

See the full <a href="https://github.com/remcofl/HilbertSFC/blob/main/benchmark-cpu.md">CPU benchmarks</a> and <a href="https://github.com/remcofl/HilbertSFC/blob/main/benchmark-gpu.md">GPU benchmarks</a>.

## When and why to use HilbertSFC?

If you have 2D or 3D coordinates and need a 1D ordering that preserves spatial locality, the Hilbert space-filling curve is a strong choice: points that are close in Euclidean space tend to remain close after mapping to a Hilbert index. HilbertSFC
is designed for high-throughput workloads, such as spatial indexing (GIS/databases), scientific computing, and machine/deep learning, where Hilbert curve mapping performance matters.
