# hilbert-bench-rs

Rust benchmark runner used in this repo to compare Rust Hilbert implementations under the same benchmark semantics as `bench/bench_cli.py`.

Current implementations:

- `fast_hilbert`
- `hilbert_2d`

## Run

From the repository root:

```powershell
cargo run --release --manifest-path bench/hilbert-bench-rs/Cargo.toml -- \
  --mode random --ndim 2 --nbits 16 --n 5000000 --seed 123 --threads 1 --trials 5 --min-time 0.25
```

Run a subset of implementations (repeat `--only` as needed):

```powershell
cargo run --release --manifest-path bench/hilbert-bench-rs/Cargo.toml -- \
  --only fast_hilbert --only hilbert_2d --mode random --nbits 16 --n 5000000
```

## Notes

- `--ndim` is currently fixed to `2` in the Rust bench.
- `--min-time` is the minimum total wall time across all trials.
- `--threads` controls Rayon worker threads (`0` = Rayon default).
- `--validate` checks `decode(encode(coords))` on `--validate-n` points.
