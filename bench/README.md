# Benchmarks

This folder contains benchmark runners for Python and Rust.

- Python runner: `bench/bench_cli.py`
- Rust runner: `bench/hilbert-bench-rs`

Both runners share the same core benchmark semantics:

- input modes: `grid`, `arange`, `random`
- throughput reporting per implementation
- median and sample stddev across trials
- `--min-time` is minimum total wall time across all trials

## Python bench CLI

Run with:

```powershell
uv run --group bench python bench/bench_cli.py ...
```

List builtin implementations for a dimension:

```powershell
uv run --group bench python bench/bench_cli.py --ndim 2 --list-impls
```

Run one or more builtin implementations:

```powershell
uv run --group bench python bench/bench_cli.py --ndim 2 --nbits 32 --n 1000000 --trials 1 --min-time 2 --threads 1 \
  --impl-name hilbertsfc --impl-name hilbertsfc-morton --impl-name hilbert-bytes
```

Load external implementations:

```powershell
uv run --group bench python bench/bench_cli.py --impl mymodule:IMPLS
uv run --group bench python bench/bench_cli.py --impl bench/hilbert_impls_2d.py:IMPLS --ndim 2
```

## Rust bench CLI

Run from repo root:

```powershell
cargo run --release --manifest-path bench/hilbert-bench-rs/Cargo.toml -- \
  --mode random --ndim 2 --nbits 16 --n 5000000 --seed 123 --threads 1 --trials 5 --min-time 0.25
```

Select implementations:

```powershell
cargo run --release --manifest-path bench/hilbert-bench-rs/Cargo.toml -- \
  --only fast_hilbert --only hilbert_2d --mode random --nbits 16 --n 5000000
```

## Quick parity notes

- Rust bench currently supports `--ndim 2` only.
- Python bench supports `--ndim 2` and `--ndim 3`.
