# hilbert-bench-rs

Standalone Rust benchmark runner that mirrors this repo's Python bench semantics (`bench/bench_cli.py`),
and benchmarks multiple Rust implementations:

- `fast_hilbert`
- `hilbert_2d`

## Usage

From the repo root:

```powershell
cargo run --release --manifest-path bench/hilbert-bench-rs/Cargo.toml -- \
  --mode random --nbits 16 --n 5000000 --seed 123 --threads 1 --trials 5 --min-time 0.25 --validate
```

Select a single implementation (repeatable):

```powershell
cargo run --release --manifest-path bench/hilbert-bench-rs/Cargo.toml -- \
  --only hilbert_2d --mode random --nbits 16 --n 5000000
```

Notes:

- `--min-time` is total time across trials (like the Python bench).
- `--threads` controls Rayon parallelism for generation/encode/decode.
