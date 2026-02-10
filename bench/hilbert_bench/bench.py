from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numba as nb
import numpy as np

Mode = Literal["grid", "arange", "random"]
RateUnit = Literal["k", "m", "auto"]


def _coord_dtype(nbits: int) -> np.dtype:
    if nbits <= 8:
        return np.dtype(np.uint8)
    if nbits <= 16:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def _index_dtype(nbits: int) -> np.dtype:
    # Back-compat for older code: 2D indices need 2*nbits bits.
    return _index_dtype_ndim(int(nbits), 2)


def _index_dtype_ndim(nbits: int, ndim: int) -> np.dtype:
    """Choose the smallest unsigned dtype that fits `ndim*nbits` bits."""
    bits = int(ndim) * int(nbits)
    if bits <= 16:
        return np.dtype(np.uint16)
    if bits <= 32:
        return np.dtype(np.uint32)
    if bits <= 64:
        return np.dtype(np.uint64)
    raise ValueError(f"Index needs {bits} bits; no builtin unsigned dtype fits")


# --- SplitMix64 (Numba-jitted) ------------------------------------------------
_SM64_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SM64_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_SM64_MUL2 = np.uint64(0x94D049BB133111EB)


@nb.njit("uint64(uint64)", inline="always", cache=True)
def splitmix64_mix(z: np.uint64) -> np.uint64:
    z ^= z >> 30
    z *= _SM64_MUL1
    z ^= z >> 27
    z *= _SM64_MUL2
    z ^= z >> 31
    return z


@nb.njit(parallel=True, cache=True)
def splitmix64_fill_xy(
    xs: np.ndarray, ys: np.ndarray, mask: np.uint64, seed: np.uint64
) -> None:
    """Fill `xs` and `ys` with SplitMix64 outputs masked to `mask`.

    Deterministic across thread counts by computing the k-th output directly:
    output(k) uses state = seed + GAMMA*(k+1)

    Two RNG outputs per point:
    - x[i] uses k=2*i
    - y[i] uses k=2*i+1
    """
    n = xs.shape[0]
    for i in nb.prange(n):
        s1 = seed + _SM64_GAMMA * np.uint64(2 * i + 1)
        xs[i] = splitmix64_mix(s1) & mask
        s2 = seed + _SM64_GAMMA * np.uint64(2 * i + 2)
        ys[i] = splitmix64_mix(s2) & mask


@nb.njit(parallel=True, cache=True)
def splitmix64_fill_xyz(
    xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, mask: np.uint64, seed: np.uint64
) -> None:
    """Fill `xs`, `ys`, and `zs` with SplitMix64 outputs masked to `mask`.

    Deterministic across thread counts by computing the k-th output directly:
    output(k) uses state = seed + GAMMA*(k+1)

    Three RNG outputs per point:
    - x[i] uses k=3*i
    - y[i] uses k=3*i+1
    - z[i] uses k=3*i+2
    """
    n = xs.shape[0]
    for i in nb.prange(n):
        s1 = seed + _SM64_GAMMA * np.uint64(3 * i + 1)
        xs[i] = splitmix64_mix(s1) & mask
        s2 = seed + _SM64_GAMMA * np.uint64(3 * i + 2)
        ys[i] = splitmix64_mix(s2) & mask
        s3 = seed + _SM64_GAMMA * np.uint64(3 * i + 3)
        zs[i] = splitmix64_mix(s3) & mask


# --- Bench API ----------------------------------------------------------------
# Some implementations return the output buffer(s); others return None.
EncodeBatch = Callable[..., object]
DecodeBatch = Callable[..., object]


@dataclass(frozen=True)
class BenchResult:
    name: str
    # Median seconds per call across trials.
    seconds: float
    # Sample standard deviation of seconds per call across trials.
    seconds_std: float
    points: int
    effective_trials: int

    # Median and stddev for throughput across trials (computed from per-trial seconds).
    mpts_per_s: float
    mpts_per_s_std: float

    @property
    def ms(self) -> float:
        return self.seconds * 1e3

    @property
    def ms_std(self) -> float:
        return self.seconds_std * 1e3

    @property
    def ns_per_point(self) -> float:
        return (self.seconds / self.points) * 1e9

    def _format_rate(self, rate_unit: RateUnit) -> tuple[float, float, str]:
        if rate_unit == "auto":
            # Keep output roughly in [1..1000) where possible.
            rate_unit = "m" if self.mpts_per_s >= 1.0 else "k"

        if rate_unit == "k":
            kpts = self.mpts_per_s * 1e3
            kpts_std = self.mpts_per_s_std * 1e3
            return kpts, kpts_std, "Kpts/s"

        # Default: Mpts/s
        return self.mpts_per_s, self.mpts_per_s_std, "Mpts/s"

    def format(self, rate_unit: RateUnit = "m") -> str:
        rate, rate_std, rate_label = self._format_rate(rate_unit)
        return (
            f"{self.name}: {self.ms:.3f} ms | {self.ns_per_point:.2f} ns/pt | "
            f"{rate:.2f} ± {rate_std:.2f} {rate_label} | "
            f"trials={self.effective_trials}"
        )

    def print(self, rate_unit: RateUnit = "m") -> None:
        print(self.format(rate_unit=rate_unit))


@dataclass(frozen=True)
class HilbertImplementation:
    name: str
    encode: EncodeBatch | None = None
    decode: DecodeBatch | None = None


@dataclass
class HilbertBenchConfig:
    mode: Mode = "random"
    ndim: int = 2
    nbits: int = 16
    n: int = 5_000_000
    seed: int = 123
    threads: int = 0  # 0 = leave default
    trials: int = 5
    # Minimum total wall time spent across all trials (0 = disabled).
    min_time_s: float = 0.25
    validate: bool = False
    validate_n: int = 10_000


def _time_trials(fn: Callable[[], None], trials: int, min_time_s: float) -> np.ndarray:
    """Return seconds_per_call for one full pass per trial.

    Runs at least `trials` trials. If `min_time_s > 0`, runs additional trials
    until the total wall time across all trials is >= `min_time_s` (capped).
    """
    target_trials = max(1, int(trials))
    min_total_s = float(min_time_s)
    max_trials = 1 << 20

    seconds_list: list[float] = []
    total_s = 0.0
    i = 0
    while (i < target_trials) or (min_total_s > 0.0 and total_s < min_total_s):
        if i >= max_trials:
            break
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        dt_s = (t1 - t0) * 1e-9
        seconds_list.append(dt_s)
        total_s += dt_s
        i += 1

    return np.asarray(seconds_list, dtype=np.float64)


class HilbertBench:
    def __init__(self, config: HilbertBenchConfig):
        self.config = config

        ndim = int(config.ndim)
        if ndim not in (2, 3):
            raise ValueError("ndim must be 2 or 3")

        nbits = int(config.nbits)
        if not (1 <= nbits <= 32):
            raise ValueError("nbits must be in [1..32]")
        # 3D indices require 3*nbits bits.
        if ndim == 3 and (ndim * nbits) > 64:
            raise ValueError(
                "For ndim=3, nbits must satisfy 3*nbits <= 64 (nbits <= 21)"
            )

        if config.threads:
            nb.set_num_threads(int(config.threads))

        self._coord_dtype = _coord_dtype(nbits)
        self._index_dtype = _index_dtype_ndim(nbits, ndim)

        self.coords = self._make_points()
        self.n = int(self.coords[0].shape[0])

        self.out = np.empty(self.n, dtype=self._index_dtype)
        self.coords2 = tuple(
            np.empty(self.n, dtype=self._coord_dtype) for _ in range(int(config.ndim))
        )

    @property
    def coord_dtype(self) -> np.dtype:
        return self._coord_dtype

    @property
    def index_dtype(self) -> np.dtype:
        return self._index_dtype

    def print_setup(self) -> None:
        c = self.config
        print("hilbert bench")
        print(f"- mode: {c.mode}")
        print(f"- ndim: {int(c.ndim)}")
        print(f"- nbits: {int(c.nbits)}")
        print(f"- n: {int(self.n):,}")
        if c.mode == "random":
            print(f"- seed: {int(c.seed)}")
        print(f"- coord dtype: {self.coord_dtype}")
        print(f"- index dtype: {self.index_dtype}")
        print(f"- numba threads: {nb.get_num_threads()}")
        print(f"- trials: {int(c.trials)}")
        print(f"- min time (total): {float(c.min_time_s)}s")

    def _make_points(self) -> tuple[np.ndarray, ...]:
        c = self.config
        ndim = int(c.ndim)
        nbits = int(c.nbits)

        if c.mode == "grid":
            side = 1 << nbits
            coords = np.arange(side, dtype=self._coord_dtype)
            if ndim == 2:
                xs = np.repeat(coords, side)
                ys = np.tile(coords, side)
                return xs, ys
            # ndim == 3
            # x changes slowest, z fastest.
            xs = np.repeat(coords, side * side)
            ys = np.tile(np.repeat(coords, side), side)
            zs = np.tile(coords, side * side)
            return xs, ys, zs

        if c.mode == "arange":
            mask = np.uint64((1 << nbits) - 1)
            i = np.arange(int(c.n), dtype=np.uint64)
            xs = (i & mask).astype(self._coord_dtype, copy=False)
            ys = ((i >> np.uint64(nbits)) & mask).astype(self._coord_dtype, copy=False)
            if ndim == 2:
                return xs, ys
            zs = ((i >> np.uint64(2 * nbits)) & mask).astype(
                self._coord_dtype, copy=False
            )
            return xs, ys, zs

        # random
        mask = np.uint64((1 << nbits) - 1)
        xs = np.empty(int(c.n), dtype=self._coord_dtype)
        ys = np.empty(int(c.n), dtype=self._coord_dtype)
        if ndim == 2:
            splitmix64_fill_xy(xs, ys, mask, np.uint64(int(c.seed)))
            return xs, ys
        zs = np.empty(int(c.n), dtype=self._coord_dtype)
        splitmix64_fill_xyz(xs, ys, zs, mask, np.uint64(int(c.seed)))
        return xs, ys, zs

    def _call_encode(
        self, encode: EncodeBatch, coords: tuple[np.ndarray, ...], out: np.ndarray
    ) -> None:
        """Call encode with either (..., out) or (...) returning an array."""
        try:
            encode(*coords, out)
            return
        except TypeError:
            res = encode(*coords)
            if isinstance(res, np.ndarray):
                out[:] = res

    def _call_decode(
        self, decode: DecodeBatch, idx: np.ndarray, coords_out: tuple[np.ndarray, ...]
    ) -> None:
        """Call decode with either (idx, ...out) or (idx) returning a tuple."""
        try:
            decode(idx, *coords_out)
            return
        except TypeError:
            res = decode(idx)
            if isinstance(res, tuple) and len(res) == len(coords_out):
                for dst, src in zip(coords_out, res):
                    dst[:] = src

    def run_one(
        self,
        impl: HilbertImplementation,
        *,
        print_setup: bool = True,
        print_results: bool = True,
        rate_unit: RateUnit = "m",
    ) -> dict[str, BenchResult]:
        if print_setup:
            self.print_setup()
            print(f"- impl: {impl.name}")

        results: dict[str, BenchResult] = {}
        c = self.config

        encode = impl.encode
        decode = impl.decode

        if c.validate and encode is not None and decode is not None:
            # Validate correctness before running any timings.
            nval = min(int(c.validate_n), self.n)
            coords = tuple(a[:nval] for a in self.coords)
            coords2 = tuple(a[:nval] for a in self.coords2)
            self._call_encode(encode, coords, self.out[:nval])
            self._call_decode(decode, self.out[:nval], coords2)
            ok = all(
                np.array_equal(src[:nval], dst[:nval])
                for src, dst in zip(coords, coords2)
            )
            if not ok:
                raise AssertionError(
                    f"Validation failed for {impl.name}: decode(encode(coords)) != coords"
                )

        if encode is not None:
            # warmup compile
            coords1 = tuple(a[:1] for a in self.coords)
            self._call_encode(encode, coords1, self.out[:1])

            def _encode_call() -> None:
                self._call_encode(encode, self.coords, self.out)

            seconds_s = _time_trials(_encode_call, c.trials, c.min_time_s)
            seconds_med = float(np.median(seconds_s))
            seconds_std = (
                float(np.std(seconds_s, ddof=1)) if seconds_s.size > 1 else 0.0
            )
            mpts_s = (self.n / seconds_s) / 1e6
            mpts_med = float(np.median(mpts_s))
            mpts_std = float(np.std(mpts_s, ddof=1)) if mpts_s.size > 1 else 0.0

            res = BenchResult(
                f"{impl.name}/encode",
                seconds_med,
                seconds_std,
                self.n,
                int(seconds_s.size),
                mpts_med,
                mpts_std,
            )
            results["encode"] = res
            if print_results:
                res.print(rate_unit=rate_unit)

        if decode is not None:
            # warmup compile
            coords2_1 = tuple(a[:1] for a in self.coords2)
            self._call_decode(decode, self.out[:1], coords2_1)

            def _decode_call() -> None:
                self._call_decode(decode, self.out, self.coords2)

            seconds_s = _time_trials(_decode_call, c.trials, c.min_time_s)
            seconds_med = float(np.median(seconds_s))
            seconds_std = (
                float(np.std(seconds_s, ddof=1)) if seconds_s.size > 1 else 0.0
            )
            mpts_s = (self.n / seconds_s) / 1e6
            mpts_med = float(np.median(mpts_s))
            mpts_std = float(np.std(mpts_s, ddof=1)) if mpts_s.size > 1 else 0.0

            res = BenchResult(
                f"{impl.name}/decode",
                seconds_med,
                seconds_std,
                self.n,
                int(seconds_s.size),
                mpts_med,
                mpts_std,
            )
            results["decode"] = res
            if print_results:
                res.print(rate_unit=rate_unit)

        return results

    def run_many(
        self,
        impls: dict[str, HilbertImplementation],
        *,
        print_setup: bool = True,
        print_results: bool = True,
        rate_unit: RateUnit = "m",
    ) -> dict[str, dict[str, BenchResult]]:
        all_results: dict[str, dict[str, BenchResult]] = {}
        first = True
        for name, impl in impls.items():
            impl2 = HilbertImplementation(
                name=impl.name or name,
                encode=impl.encode,
                decode=impl.decode,
            )
            if print_setup and first:
                # print setup once
                self.print_setup()
                first = False
            if print_results:
                print(f"\nimpl: {impl2.name}")
            all_results[name] = self.run_one(
                impl2,
                print_setup=False,
                print_results=print_results,
                rate_unit=rate_unit,
            )
        return all_results
