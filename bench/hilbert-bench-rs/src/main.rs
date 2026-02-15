use std::env;
use std::time::{Duration, Instant};

use rayon::prelude::*;

use hilbert_2d::Variant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Grid,
    Arange,
    Random,
}

#[derive(Debug)]
struct Args {
    mode: Mode,
    ndim: u32,
    nbits: u32,
    n: usize,
    seed: u64,
    threads: usize,
    trials: usize,
    min_time_s: f64,
    validate: bool,
    validate_n: usize,
    no_setup: bool,
    quiet: bool,
    only: Vec<String>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode: Mode::Random,
            ndim: 2,
            nbits: 16,
            n: 5_000_000,
            seed: 123,
            threads: 1,
            trials: 5,
            min_time_s: 0.25,
            validate: false,
            validate_n: 10_000,
            no_setup: false,
            quiet: false,
            only: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Implementation {
    FastHilbert,
    Hilbert2d,
}

impl Implementation {
    const ALL: [Implementation; 2] = [Implementation::FastHilbert, Implementation::Hilbert2d];

    fn id(self) -> &'static str {
        match self {
            Implementation::FastHilbert => "fast_hilbert",
            Implementation::Hilbert2d => "hilbert_2d",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "fast_hilbert" => Some(Implementation::FastHilbert),
            "hilbert_2d" => Some(Implementation::Hilbert2d),
            _ => None,
        }
    }
}

fn usage() -> &'static str {
        r#"hilbert-bench-rs (fast_hilbert, hilbert_2d)

Mirrors this repo's Python bench semantics (bench/bench_cli.py):
- modes: grid|arange|random
- trials: N full-pass trials
- min-time: minimum *total* wall time across all trials
- reports median + sample stddev across trials

Usage:
    cargo run --release --manifest-path bench/hilbert-bench-rs/Cargo.toml -- [options]

Options:
  --mode <grid|arange|random>   Input generation mode (default: random)
    --ndim <2>                     Number of dimensions (only 2 supported) (default: 2)
  --nbits <1..32>               Bits per coordinate (default: 16)
  --n <N>                       Number of points for arange/random (default: 5000000)
  --seed <SEED>                 RNG seed for random (default: 123)
    --threads <N>                 Rayon worker threads (0 = default) (default: 1)
  --trials <N>                  Minimum number of trials (default: 5)
  --min-time <seconds>          Minimum total seconds across all trials (0 disables) (default: 0.25)
    --only <impl>                  Only run selected implementation(s) (repeatable). Values: fast_hilbert, hilbert_2d
  --validate                    Check decode(encode(coords)) for first --validate-n points
  --validate-n <N>              Validation points (default: 10000)
  --no-setup                    Do not print setup header
  --quiet                       Do not print per-kernel results
  --help                        Print help

Notes:
- grid mode generates full 2^nbits x 2^nbits coordinates; it is refused for nbits > 16.
- random mode uses SplitMix64 with direct-k indexing (deterministic across thread counts), matching the Python bench.
"#
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();

    let mut it = env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--help" | "-h" => return Err("__HELP__".to_string()),
            "--validate" => args.validate = true,
            "--no-setup" => args.no_setup = true,
            "--quiet" => args.quiet = true,
            "--only" => {
                let v = it
                    .next()
                    .ok_or_else(|| "--only requires a value".to_string())?;
                args.only.push(v);
            }
            "--mode" => {
                let v = it
                    .next()
                    .ok_or_else(|| "--mode requires a value".to_string())?;
                args.mode = match v.as_str() {
                    "grid" => Mode::Grid,
                    "arange" => Mode::Arange,
                    "random" => Mode::Random,
                    _ => return Err(format!("invalid --mode: {v}")),
                };
            }
            "--ndim" => {
                let v = it
                    .next()
                    .ok_or_else(|| "--ndim requires a value".to_string())?;
                args.ndim = v
                    .parse::<u32>()
                    .map_err(|_| format!("invalid --ndim: {v}"))?;
            }
            "--nbits" => {
                let v = it
                    .next()
                    .ok_or_else(|| "--nbits requires a value".to_string())?;
                args.nbits = v
                    .parse::<u32>()
                    .map_err(|_| format!("invalid --nbits: {v}"))?;
            }
            "--n" => {
                let v = it.next().ok_or_else(|| "--n requires a value".to_string())?;
                args.n = v
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --n: {v}"))?;
            }
            "--seed" => {
                let v = it
                    .next()
                    .ok_or_else(|| "--seed requires a value".to_string())?;
                args.seed = v
                    .parse::<u64>()
                    .map_err(|_| format!("invalid --seed: {v}"))?;
            }
            "--threads" => {
                let v = it
                    .next()
                    .ok_or_else(|| "--threads requires a value".to_string())?;
                args.threads = v
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --threads: {v}"))?;
            }
            "--trials" => {
                let v = it
                    .next()
                    .ok_or_else(|| "--trials requires a value".to_string())?;
                args.trials = v
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --trials: {v}"))?;
            }
            "--min-time" => {
                let v = it
                    .next()
                    .ok_or_else(|| "--min-time requires a value".to_string())?;
                args.min_time_s = v
                    .parse::<f64>()
                    .map_err(|_| format!("invalid --min-time: {v}"))?;
            }
            "--validate-n" => {
                let v = it
                    .next()
                    .ok_or_else(|| "--validate-n requires a value".to_string())?;
                args.validate_n = v
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --validate-n: {v}"))?;
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    if args.ndim != 2 {
        return Err("Only --ndim 2 is supported".to_string());
    }
    if !(1..=32).contains(&args.nbits) {
        return Err("--nbits must be in [1..32]".to_string());
    }
    if args.trials == 0 {
        return Err("--trials must be >= 1".to_string());
    }
    if !(args.min_time_s.is_finite() && args.min_time_s >= 0.0) {
        return Err("--min-time must be a finite non-negative number".to_string());
    }

    Ok(args)
}

fn selected_impls(args: &Args) -> Result<Vec<Implementation>, String> {
    if args.only.is_empty() {
        return Ok(Implementation::ALL.to_vec());
    }
    let mut out: Vec<Implementation> = Vec::new();
    for s in &args.only {
        let Some(impl_) = Implementation::parse(s.as_str()) else {
            return Err(format!(
                "invalid --only: {s} (expected one of: fast_hilbert, hilbert_2d)"
            ));
        };
        if !out.contains(&impl_) {
            out.push(impl_);
        }
    }

    if usize::BITS < 64 && args.nbits > 16 && out.contains(&Implementation::Hilbert2d) {
        return Err(
            "hilbert_2d uses pointer-sized integers; on 32-bit targets only nbits<=16 is supported"
                .to_string(),
        );
    }

    Ok(out)
}

fn coord_dtype_name(nbits: u32) -> &'static str {
    if nbits <= 8 {
        "u8"
    } else if nbits <= 16 {
        "u16"
    } else {
        "u32"
    }
}

fn index_dtype_name_2d(nbits: u32) -> &'static str {
    let bits = 2 * nbits;
    if bits <= 16 {
        "u16"
    } else if bits <= 32 {
        "u32"
    } else {
        "u64"
    }
}

// --- SplitMix64 (matches Python: state = seed + GAMMA*(k+1), then mix) --------
const SM64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SM64_MUL1: u64 = 0xBF58_476D_1CE4_E5B9;
const SM64_MUL2: u64 = 0x94D0_49BB_1331_11EB;

#[inline(always)]
fn splitmix64_mix(mut z: u64) -> u64 {
    z ^= z >> 30;
    z = z.wrapping_mul(SM64_MUL1);
    z ^= z >> 27;
    z = z.wrapping_mul(SM64_MUL2);
    z ^= z >> 31;
    z
}

fn fill_random_xy_u32(xs: &mut [u32], ys: &mut [u32], mask: u64, seed: u64) {
    xs.par_iter_mut().zip(ys.par_iter_mut()).enumerate().for_each(|(i, (x, y))| {
        let s1 = seed.wrapping_add(SM64_GAMMA.wrapping_mul((2 * i + 1) as u64));
        *x = (splitmix64_mix(s1) & mask) as u32;
        let s2 = seed.wrapping_add(SM64_GAMMA.wrapping_mul((2 * i + 2) as u64));
        *y = (splitmix64_mix(s2) & mask) as u32;
    });
}

fn fill_random_xy_u16(xs: &mut [u16], ys: &mut [u16], mask: u64, seed: u64) {
    xs.par_iter_mut().zip(ys.par_iter_mut()).enumerate().for_each(|(i, (x, y))| {
        let s1 = seed.wrapping_add(SM64_GAMMA.wrapping_mul((2 * i + 1) as u64));
        *x = (splitmix64_mix(s1) & mask) as u16;
        let s2 = seed.wrapping_add(SM64_GAMMA.wrapping_mul((2 * i + 2) as u64));
        *y = (splitmix64_mix(s2) & mask) as u16;
    });
}

fn fill_random_xy_u8(xs: &mut [u8], ys: &mut [u8], mask: u64, seed: u64) {
    xs.par_iter_mut().zip(ys.par_iter_mut()).enumerate().for_each(|(i, (x, y))| {
        let s1 = seed.wrapping_add(SM64_GAMMA.wrapping_mul((2 * i + 1) as u64));
        *x = (splitmix64_mix(s1) & mask) as u8;
        let s2 = seed.wrapping_add(SM64_GAMMA.wrapping_mul((2 * i + 2) as u64));
        *y = (splitmix64_mix(s2) & mask) as u8;
    });
}

// --- Stats -------------------------------------------------------------------
fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

fn sample_stddev(xs: &[f64]) -> f64 {
    if xs.len() <= 1 {
        return 0.0;
    }
    let mean = xs.iter().sum::<f64>() / (xs.len() as f64);
    let var = xs.iter().map(|v| {
        let d = v - mean;
        d * d
    }).sum::<f64>() / ((xs.len() - 1) as f64);
    var.sqrt()
}

fn time_trials(mut f: impl FnMut(), trials: usize, min_total: Duration) -> Vec<f64> {
    let target_trials = trials.max(1);
    let mut seconds = Vec::with_capacity(target_trials);
    let mut total = Duration::from_secs(0);
    let mut i = 0usize;
    let max_trials = 1usize << 20;

    while (i < target_trials) || (min_total > Duration::from_secs(0) && total < min_total) {
        if i >= max_trials {
            break;
        }
        let t0 = Instant::now();
        f();
        let dt = t0.elapsed();
        seconds.push(dt.as_secs_f64());
        total += dt;
        i += 1;
    }

    seconds
}

fn print_setup(args: &Args, n: usize) {
    println!("hilbert bench");
    match args.mode {
        Mode::Grid => println!("- mode: grid"),
        Mode::Arange => println!("- mode: arange"),
        Mode::Random => println!("- mode: random"),
    }
    println!("- ndim: {}", args.ndim);
    println!("- nbits: {}", args.nbits);
    println!("- n: {n}");
    if args.mode == Mode::Random {
        println!("- seed: {}", args.seed);
    }
    println!("- coord dtype: {}", coord_dtype_name(args.nbits));
    println!("- index dtype: {}", index_dtype_name_2d(args.nbits));
    println!("- threads: {}", if args.threads == 0 { rayon::current_num_threads() } else { args.threads });
    println!("- trials: {}", args.trials);
    println!("- min time (total): {}s", args.min_time_s);
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) if e == "__HELP__" => {
            eprintln!("{}", usage());
            return;
        }
        Err(e) => {
            eprintln!("error: {e}\n\n{}", usage());
            std::process::exit(2);
        }
    };

    let impls = match selected_impls(&args) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}\n\n{}", usage());
            std::process::exit(2);
        }
    };

    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .unwrap();
    }

    let min_total = Duration::from_secs_f64(args.min_time_s);
    let order = args.nbits as u8;

    // Select types to match Python bench dtypes.
    if args.nbits <= 8 {
        run_bench_u8(&args, &impls, order, min_total);
    } else if args.nbits <= 16 {
        run_bench_u16(&args, &impls, order, min_total);
    } else {
        run_bench_u32(&args, &impls, order, min_total);
    }
}

fn make_points_grid_u8(nbits: u32) -> Result<(Vec<u8>, Vec<u8>), String> {
    if nbits > 8 {
        return Err("grid mode with u8 coords requires nbits <= 8".to_string());
    }
    let side: usize = 1usize
        .checked_shl(nbits)
        .ok_or_else(|| "side overflow".to_string())?;
    let n: usize = side
        .checked_mul(side)
        .ok_or_else(|| "grid point count overflow".to_string())?;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for x in 0..side {
        for y in 0..side {
            xs.push(x as u8);
            ys.push(y as u8);
        }
    }
    Ok((xs, ys))
}

fn make_points_grid_u16(nbits: u32) -> Result<(Vec<u16>, Vec<u16>), String> {
    if nbits > 16 {
        return Err("grid mode is refused for nbits > 16 (would allocate an enormous grid)".to_string());
    }
    let side: usize = 1usize
        .checked_shl(nbits)
        .ok_or_else(|| "side overflow".to_string())?;
    let n: usize = side
        .checked_mul(side)
        .ok_or_else(|| "grid point count overflow".to_string())?;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for x in 0..side {
        for y in 0..side {
            xs.push(x as u16);
            ys.push(y as u16);
        }
    }
    Ok((xs, ys))
}

fn make_points_grid_u32(nbits: u32) -> Result<(Vec<u32>, Vec<u32>), String> {
    if nbits > 16 {
        return Err("grid mode is refused for nbits > 16 (would allocate an enormous grid)".to_string());
    }
    let side: usize = 1usize
        .checked_shl(nbits)
        .ok_or_else(|| "side overflow".to_string())?;
    let n: usize = side
        .checked_mul(side)
        .ok_or_else(|| "grid point count overflow".to_string())?;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for x in 0..side {
        for y in 0..side {
            xs.push(x as u32);
            ys.push(y as u32);
        }
    }
    Ok((xs, ys))
}

fn make_points_arange_u8(nbits: u32, n: usize) -> (Vec<u8>, Vec<u8>) {
    let mask: u64 = (1u64 << nbits) - 1;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..(n as u64) {
        xs.push((i & mask) as u8);
        ys.push(((i >> nbits) & mask) as u8);
    }
    (xs, ys)
}

fn make_points_arange_u16(nbits: u32, n: usize) -> (Vec<u16>, Vec<u16>) {
    let mask: u64 = (1u64 << nbits) - 1;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..(n as u64) {
        xs.push((i & mask) as u16);
        ys.push(((i >> nbits) & mask) as u16);
    }
    (xs, ys)
}

fn make_points_arange_u32(nbits: u32, n: usize) -> (Vec<u32>, Vec<u32>) {
    let mask: u64 = (1u64 << nbits) - 1;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..(n as u64) {
        xs.push((i & mask) as u32);
        ys.push(((i >> nbits) & mask) as u32);
    }
    (xs, ys)
}

fn make_points_random_u8(nbits: u32, n: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
    let mask: u64 = (1u64 << nbits) - 1;
    let mut xs = vec![0u8; n];
    let mut ys = vec![0u8; n];
    fill_random_xy_u8(&mut xs, &mut ys, mask, seed);
    (xs, ys)
}

fn make_points_random_u16(nbits: u32, n: usize, seed: u64) -> (Vec<u16>, Vec<u16>) {
    let mask: u64 = (1u64 << nbits) - 1;
    let mut xs = vec![0u16; n];
    let mut ys = vec![0u16; n];
    fill_random_xy_u16(&mut xs, &mut ys, mask, seed);
    (xs, ys)
}

fn make_points_random_u32(nbits: u32, n: usize, seed: u64) -> (Vec<u32>, Vec<u32>) {
    let mask: u64 = (1u64 << nbits) - 1;
    let mut xs = vec![0u32; n];
    let mut ys = vec![0u32; n];
    fill_random_xy_u32(&mut xs, &mut ys, mask, seed);
    (xs, ys)
}

fn run_bench_u8(args: &Args, impls: &[Implementation], order: u8, min_total: Duration) {
    let (xs, ys): (Vec<u8>, Vec<u8>) = match args.mode {
        Mode::Grid => match make_points_grid_u8(args.nbits) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: {e}\n\n{}", usage());
                std::process::exit(2);
            }
        },
        Mode::Arange => make_points_arange_u8(args.nbits, args.n),
        Mode::Random => make_points_random_u8(args.nbits, args.n, args.seed),
    };

    let n = xs.len();

    let mut out = vec![0u16; n];
    let mut xs2 = vec![0u8; n];
    let mut ys2 = vec![0u8; n];

    if !args.no_setup {
        print_setup(args, n);
    }

    for impl_ in impls {
        if !args.no_setup {
            println!("\nimpl: {}", impl_.id());
        }

        if n > 0 {
            encode_pass_u8_impl(*impl_, &xs[..1], &ys[..1], &mut out[..1], order);
            decode_pass_u8_impl(*impl_, &out[..1], &mut xs2[..1], &mut ys2[..1], order);
        }

        if args.validate {
            let check_n = args.validate_n.min(n);
            encode_pass_u8_impl(*impl_, &xs[..check_n], &ys[..check_n], &mut out[..check_n], order);
            decode_pass_u8_impl(*impl_, &out[..check_n], &mut xs2[..check_n], &mut ys2[..check_n], order);
            if xs[..check_n] != xs2[..check_n] || ys[..check_n] != ys2[..check_n] {
                eprintln!(
                    "Validation failed for {}: decode(encode(coords)) did not match",
                    impl_.id()
                );
                std::process::exit(1);
            }
        }

        let enc_seconds = time_trials(
            || {
                encode_pass_u8_impl(*impl_, &xs, &ys, &mut out, order);
            },
            args.trials,
            min_total,
        );
        let enc_med = median(enc_seconds.clone());
        let enc_std = sample_stddev(&enc_seconds);
        let enc_mpts: Vec<f64> = enc_seconds.iter().map(|s| (n as f64 / *s) / 1e6).collect();
        let enc_mpts_med = median(enc_mpts.clone());
        let enc_mpts_std = sample_stddev(&enc_mpts);
        if !args.quiet {
            print_result(
                &format!("{}/encode", impl_.id()),
                enc_med,
                enc_std,
                n,
                enc_seconds.len(),
                enc_mpts_med,
                enc_mpts_std,
            );
        }

        // prepare indices for decode
        encode_pass_u8_impl(*impl_, &xs, &ys, &mut out, order);

        let dec_seconds = time_trials(
            || {
                decode_pass_u8_impl(*impl_, &out, &mut xs2, &mut ys2, order);
            },
            args.trials,
            min_total,
        );
        let dec_med = median(dec_seconds.clone());
        let dec_std = sample_stddev(&dec_seconds);
        let dec_mpts: Vec<f64> = dec_seconds.iter().map(|s| (n as f64 / *s) / 1e6).collect();
        let dec_mpts_med = median(dec_mpts.clone());
        let dec_mpts_std = sample_stddev(&dec_mpts);
        if !args.quiet {
            print_result(
                &format!("{}/decode", impl_.id()),
                dec_med,
                dec_std,
                n,
                dec_seconds.len(),
                dec_mpts_med,
                dec_mpts_std,
            );
        }
    }
}

fn run_bench_u16(args: &Args, impls: &[Implementation], order: u8, min_total: Duration) {
    let (xs, ys): (Vec<u16>, Vec<u16>) = match args.mode {
        Mode::Grid => match make_points_grid_u16(args.nbits) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: {e}\n\n{}", usage());
                std::process::exit(2);
            }
        },
        Mode::Arange => make_points_arange_u16(args.nbits, args.n),
        Mode::Random => make_points_random_u16(args.nbits, args.n, args.seed),
    };

    let n = xs.len();

    let mut out = vec![0u32; n];
    let mut xs2 = vec![0u16; n];
    let mut ys2 = vec![0u16; n];

    if !args.no_setup {
        print_setup(args, n);
    }

    for impl_ in impls {
        if !args.no_setup {
            println!("\nimpl: {}", impl_.id());
        }

        if n > 0 {
            encode_pass_u16_impl(*impl_, &xs[..1], &ys[..1], &mut out[..1], order);
            decode_pass_u16_impl(*impl_, &out[..1], &mut xs2[..1], &mut ys2[..1], order);
        }

        if args.validate {
            let check_n = args.validate_n.min(n);
            encode_pass_u16_impl(*impl_, &xs[..check_n], &ys[..check_n], &mut out[..check_n], order);
            decode_pass_u16_impl(*impl_, &out[..check_n], &mut xs2[..check_n], &mut ys2[..check_n], order);
            if xs[..check_n] != xs2[..check_n] || ys[..check_n] != ys2[..check_n] {
                eprintln!(
                    "Validation failed for {}: decode(encode(coords)) did not match",
                    impl_.id()
                );
                std::process::exit(1);
            }
        }

        let enc_seconds = time_trials(
            || {
                encode_pass_u16_impl(*impl_, &xs, &ys, &mut out, order);
            },
            args.trials,
            min_total,
        );
        let enc_med = median(enc_seconds.clone());
        let enc_std = sample_stddev(&enc_seconds);
        let enc_mpts: Vec<f64> = enc_seconds.iter().map(|s| (n as f64 / *s) / 1e6).collect();
        let enc_mpts_med = median(enc_mpts.clone());
        let enc_mpts_std = sample_stddev(&enc_mpts);
        if !args.quiet {
            print_result(
                &format!("{}/encode", impl_.id()),
                enc_med,
                enc_std,
                n,
                enc_seconds.len(),
                enc_mpts_med,
                enc_mpts_std,
            );
        }

        encode_pass_u16_impl(*impl_, &xs, &ys, &mut out, order);

        let dec_seconds = time_trials(
            || {
                decode_pass_u16_impl(*impl_, &out, &mut xs2, &mut ys2, order);
            },
            args.trials,
            min_total,
        );
        let dec_med = median(dec_seconds.clone());
        let dec_std = sample_stddev(&dec_seconds);
        let dec_mpts: Vec<f64> = dec_seconds.iter().map(|s| (n as f64 / *s) / 1e6).collect();
        let dec_mpts_med = median(dec_mpts.clone());
        let dec_mpts_std = sample_stddev(&dec_mpts);
        if !args.quiet {
            print_result(
                &format!("{}/decode", impl_.id()),
                dec_med,
                dec_std,
                n,
                dec_seconds.len(),
                dec_mpts_med,
                dec_mpts_std,
            );
        }
    }
}

fn run_bench_u32(args: &Args, impls: &[Implementation], order: u8, min_total: Duration) {
    let (xs, ys): (Vec<u32>, Vec<u32>) = match args.mode {
        Mode::Grid => match make_points_grid_u32(args.nbits) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: {e}\n\n{}", usage());
                std::process::exit(2);
            }
        },
        Mode::Arange => make_points_arange_u32(args.nbits, args.n),
        Mode::Random => make_points_random_u32(args.nbits, args.n, args.seed),
    };

    let n = xs.len();

    let mut out = vec![0u64; n];
    let mut xs2 = vec![0u32; n];
    let mut ys2 = vec![0u32; n];

    if !args.no_setup {
        print_setup(args, n);
    }

    for impl_ in impls {
        if !args.no_setup {
            println!("\nimpl: {}", impl_.id());
        }

        if n > 0 {
            encode_pass_u32_impl(*impl_, &xs[..1], &ys[..1], &mut out[..1], order);
            decode_pass_u32_impl(*impl_, &out[..1], &mut xs2[..1], &mut ys2[..1], order);
        }

        if args.validate {
            let check_n = args.validate_n.min(n);
            encode_pass_u32_impl(*impl_, &xs[..check_n], &ys[..check_n], &mut out[..check_n], order);
            decode_pass_u32_impl(*impl_, &out[..check_n], &mut xs2[..check_n], &mut ys2[..check_n], order);
            if xs[..check_n] != xs2[..check_n] || ys[..check_n] != ys2[..check_n] {
                eprintln!(
                    "Validation failed for {}: decode(encode(coords)) did not match",
                    impl_.id()
                );
                std::process::exit(1);
            }
        }

        let enc_seconds = time_trials(
            || {
                encode_pass_u32_impl(*impl_, &xs, &ys, &mut out, order);
            },
            args.trials,
            min_total,
        );
        let enc_med = median(enc_seconds.clone());
        let enc_std = sample_stddev(&enc_seconds);
        let enc_mpts: Vec<f64> = enc_seconds.iter().map(|s| (n as f64 / *s) / 1e6).collect();
        let enc_mpts_med = median(enc_mpts.clone());
        let enc_mpts_std = sample_stddev(&enc_mpts);
        if !args.quiet {
            print_result(
                &format!("{}/encode", impl_.id()),
                enc_med,
                enc_std,
                n,
                enc_seconds.len(),
                enc_mpts_med,
                enc_mpts_std,
            );
        }

        encode_pass_u32_impl(*impl_, &xs, &ys, &mut out, order);

        let dec_seconds = time_trials(
            || {
                decode_pass_u32_impl(*impl_, &out, &mut xs2, &mut ys2, order);
            },
            args.trials,
            min_total,
        );
        let dec_med = median(dec_seconds.clone());
        let dec_std = sample_stddev(&dec_seconds);
        let dec_mpts: Vec<f64> = dec_seconds.iter().map(|s| (n as f64 / *s) / 1e6).collect();
        let dec_mpts_med = median(dec_mpts.clone());
        let dec_mpts_std = sample_stddev(&dec_mpts);
        if !args.quiet {
            print_result(
                &format!("{}/decode", impl_.id()),
                dec_med,
                dec_std,
                n,
                dec_seconds.len(),
                dec_mpts_med,
                dec_mpts_std,
            );
        }
    }
}

fn print_result(
    name: &str,
    seconds_med: f64,
    _seconds_std: f64,
    points: usize,
    trials: usize,
    mpts_med: f64,
    mpts_std: f64,
) {
    let ms = seconds_med * 1e3;
    let ns_per_pt = (seconds_med / (points as f64)) * 1e9;
    println!(
        "{name}: {ms:.3} ms | {ns_per_pt:.2} ns/pt | {mpts_med:.2} ± {mpts_std:.2} Mpts/s | trials={trials}"
    );
}


fn encode_pass_u8_impl(impl_: Implementation, xs: &[u8], ys: &[u8], out: &mut [u16], order: u8) {
    match impl_ {
        Implementation::FastHilbert => encode_pass_u8(xs, ys, out, order),
        Implementation::Hilbert2d => {
            assert_eq!(xs.len(), ys.len());
            assert_eq!(xs.len(), out.len());
            if rayon::current_num_threads() > 1 {
                out.par_iter_mut().enumerate().for_each(|(i, dst)| {
                    *dst = hilbert_2d::xy2h_discrete(
                        xs[i] as usize,
                        ys[i] as usize,
                        order as usize,
                        Variant::Hilbert,
                    ) as u16;
                });
            } else {
                for i in 0..xs.len() {
                    out[i] = hilbert_2d::xy2h_discrete(
                        xs[i] as usize,
                        ys[i] as usize,
                        order as usize,
                        Variant::Hilbert,
                    ) as u16;
                }
            }
        }
    }
}

fn encode_pass_u16_impl(
    impl_: Implementation,
    xs: &[u16],
    ys: &[u16],
    out: &mut [u32],
    order: u8,
) {
    match impl_ {
        Implementation::FastHilbert => encode_pass_u16(xs, ys, out, order),
        Implementation::Hilbert2d => {
            assert_eq!(xs.len(), ys.len());
            assert_eq!(xs.len(), out.len());
            if rayon::current_num_threads() > 1 {
                out.par_iter_mut().enumerate().for_each(|(i, dst)| {
                    *dst = hilbert_2d::xy2h_discrete(
                        xs[i] as usize,
                        ys[i] as usize,
                        order as usize,
                        Variant::Hilbert,
                    ) as u32;
                });
            } else {
                for i in 0..xs.len() {
                    out[i] = hilbert_2d::xy2h_discrete(
                        xs[i] as usize,
                        ys[i] as usize,
                        order as usize,
                        Variant::Hilbert,
                    ) as u32;
                }
            }
        }
    }
}

fn encode_pass_u32_impl(
    impl_: Implementation,
    xs: &[u32],
    ys: &[u32],
    out: &mut [u64],
    order: u8,
) {
    match impl_ {
        Implementation::FastHilbert => encode_pass_u32(xs, ys, out, order),
        Implementation::Hilbert2d => {
            assert_eq!(xs.len(), ys.len());
            assert_eq!(xs.len(), out.len());
            if rayon::current_num_threads() > 1 {
                out.par_iter_mut().enumerate().for_each(|(i, dst)| {
                    *dst = hilbert_2d::xy2h_discrete(
                        xs[i] as usize,
                        ys[i] as usize,
                        order as usize,
                        Variant::Hilbert,
                    ) as u64;
                });
            } else {
                for i in 0..xs.len() {
                    out[i] = hilbert_2d::xy2h_discrete(
                        xs[i] as usize,
                        ys[i] as usize,
                        order as usize,
                        Variant::Hilbert,
                    ) as u64;
                }
            }
        }
    }
}

fn decode_pass_u8_impl(
    impl_: Implementation,
    idx: &[u16],
    xs2: &mut [u8],
    ys2: &mut [u8],
    order: u8,
) {
    match impl_ {
        Implementation::FastHilbert => decode_pass_u8(idx, xs2, ys2, order),
        Implementation::Hilbert2d => {
            assert_eq!(idx.len(), xs2.len());
            assert_eq!(idx.len(), ys2.len());
            if rayon::current_num_threads() > 1 {
                xs2.par_iter_mut()
                    .zip(ys2.par_iter_mut())
                    .enumerate()
                    .for_each(|(i, (x, y))| {
                        let (xx, yy) = hilbert_2d::h2xy_discrete(
                            idx[i] as usize,
                            order as usize,
                            Variant::Hilbert,
                        );
                        *x = xx as u8;
                        *y = yy as u8;
                    });
            } else {
                for i in 0..idx.len() {
                    let (x, y) = hilbert_2d::h2xy_discrete(
                        idx[i] as usize,
                        order as usize,
                        Variant::Hilbert,
                    );
                    xs2[i] = x as u8;
                    ys2[i] = y as u8;
                }
            }
        }
    }
}

fn decode_pass_u16_impl(
    impl_: Implementation,
    idx: &[u32],
    xs2: &mut [u16],
    ys2: &mut [u16],
    order: u8,
) {
    match impl_ {
        Implementation::FastHilbert => decode_pass_u16(idx, xs2, ys2, order),
        Implementation::Hilbert2d => {
            assert_eq!(idx.len(), xs2.len());
            assert_eq!(idx.len(), ys2.len());
            if rayon::current_num_threads() > 1 {
                xs2.par_iter_mut()
                    .zip(ys2.par_iter_mut())
                    .enumerate()
                    .for_each(|(i, (x, y))| {
                        let (xx, yy) = hilbert_2d::h2xy_discrete(
                            idx[i] as usize,
                            order as usize,
                            Variant::Hilbert,
                        );
                        *x = xx as u16;
                        *y = yy as u16;
                    });
            } else {
                for i in 0..idx.len() {
                    let (x, y) = hilbert_2d::h2xy_discrete(
                        idx[i] as usize,
                        order as usize,
                        Variant::Hilbert,
                    );
                    xs2[i] = x as u16;
                    ys2[i] = y as u16;
                }
            }
        }
    }
}

fn decode_pass_u32_impl(
    impl_: Implementation,
    idx: &[u64],
    xs2: &mut [u32],
    ys2: &mut [u32],
    order: u8,
) {
    match impl_ {
        Implementation::FastHilbert => decode_pass_u32(idx, xs2, ys2, order),
        Implementation::Hilbert2d => {
            assert_eq!(idx.len(), xs2.len());
            assert_eq!(idx.len(), ys2.len());
            if rayon::current_num_threads() > 1 {
                xs2.par_iter_mut()
                    .zip(ys2.par_iter_mut())
                    .enumerate()
                    .for_each(|(i, (x, y))| {
                        let (xx, yy) = hilbert_2d::h2xy_discrete(
                            idx[i] as usize,
                            order as usize,
                            Variant::Hilbert,
                        );
                        *x = xx as u32;
                        *y = yy as u32;
                    });
            } else {
                for i in 0..idx.len() {
                    let (x, y) = hilbert_2d::h2xy_discrete(
                        idx[i] as usize,
                        order as usize,
                        Variant::Hilbert,
                    );
                    xs2[i] = x as u32;
                    ys2[i] = y as u32;
                }
            }
        }
    }
}


fn encode_pass_u8(xs: &[u8], ys: &[u8], out: &mut [u16], order: u8) {
    assert_eq!(xs.len(), ys.len());
    assert_eq!(xs.len(), out.len());
    if rayon::current_num_threads() > 1 {
        out.par_iter_mut().enumerate().for_each(|(i, dst)| {
            *dst = fast_hilbert::xy2h(xs[i], ys[i], order);
        });
    } else {
        for i in 0..xs.len() {
            out[i] = fast_hilbert::xy2h(xs[i], ys[i], order);
        }
    }
}

fn encode_pass_u16(xs: &[u16], ys: &[u16], out: &mut [u32], order: u8) {
    assert_eq!(xs.len(), ys.len());
    assert_eq!(xs.len(), out.len());
    if rayon::current_num_threads() > 1 {
        out.par_iter_mut().enumerate().for_each(|(i, dst)| {
            *dst = fast_hilbert::xy2h(xs[i], ys[i], order);
        });
    } else {
        for i in 0..xs.len() {
            out[i] = fast_hilbert::xy2h(xs[i], ys[i], order);
        }
    }
}

fn encode_pass_u32(xs: &[u32], ys: &[u32], out: &mut [u64], order: u8) {
    assert_eq!(xs.len(), ys.len());
    assert_eq!(xs.len(), out.len());
    if rayon::current_num_threads() > 1 {
        out.par_iter_mut().enumerate().for_each(|(i, dst)| {
            *dst = fast_hilbert::xy2h(xs[i], ys[i], order);
        });
    } else {
        for i in 0..xs.len() {
            out[i] = fast_hilbert::xy2h(xs[i], ys[i], order);
        }
    }
}

fn decode_pass_u8(idx: &[u16], xs2: &mut [u8], ys2: &mut [u8], order: u8) {
    assert_eq!(idx.len(), xs2.len());
    assert_eq!(idx.len(), ys2.len());
    if rayon::current_num_threads() > 1 {
        xs2.par_iter_mut()
            .zip(ys2.par_iter_mut())
            .enumerate()
            .for_each(|(i, (x, y))| {
                let (xx, yy) = fast_hilbert::h2xy::<u8>(idx[i], order);
                *x = xx;
                *y = yy;
            });
    } else {
        for i in 0..idx.len() {
            let (x, y) = fast_hilbert::h2xy::<u8>(idx[i], order);
            xs2[i] = x;
            ys2[i] = y;
        }
    }
}

fn decode_pass_u16(idx: &[u32], xs2: &mut [u16], ys2: &mut [u16], order: u8) {
    assert_eq!(idx.len(), xs2.len());
    assert_eq!(idx.len(), ys2.len());
    if rayon::current_num_threads() > 1 {
        xs2.par_iter_mut()
            .zip(ys2.par_iter_mut())
            .enumerate()
            .for_each(|(i, (x, y))| {
                let (xx, yy) = fast_hilbert::h2xy::<u16>(idx[i], order);
                *x = xx;
                *y = yy;
            });
    } else {
        for i in 0..idx.len() {
            let (x, y) = fast_hilbert::h2xy::<u16>(idx[i], order);
            xs2[i] = x;
            ys2[i] = y;
        }
    }
}

fn decode_pass_u32(idx: &[u64], xs2: &mut [u32], ys2: &mut [u32], order: u8) {
    assert_eq!(idx.len(), xs2.len());
    assert_eq!(idx.len(), ys2.len());
    if rayon::current_num_threads() > 1 {
        xs2.par_iter_mut()
            .zip(ys2.par_iter_mut())
            .enumerate()
            .for_each(|(i, (x, y))| {
                let (xx, yy) = fast_hilbert::h2xy::<u32>(idx[i], order);
                *x = xx;
                *y = yy;
            });
    } else {
        for i in 0..idx.len() {
            let (x, y) = fast_hilbert::h2xy::<u32>(idx[i], order);
            xs2[i] = x;
            ys2[i] = y;
        }
    }
}
