import os
import time

import torch
import triton
from triton import runtime

from hilbertsfc.torch import (
    hilbert_decode_2d,
    hilbert_decode_3d,
    hilbert_encode_2d,
    hilbert_encode_3d,
    precache_compile_luts,
)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    torch.manual_seed(0)

    # When enabled, HilbertSFC prints per-config Triton autotune timings.
    # The kernels key on n_elements, so you get one print per tested size.
    print_autotune_config_timings = True
    if print_autotune_config_timings:
        os.environ["HILBERTSFC_TRITON_PRINT_ALL_AUTOTUNE_CONFIGS"] = "1"

    # Tweak these manually while experimenting.
    device = torch.device(
        "cuda"
    )  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dims = 2
    nbits = 2
    direction = "decode"  # "encode" or "decode"
    cpu_backend = "torch"
    gpu_backend = "triton"
    dtype = torch.int32
    out_dtype = torch.int64
    use_out = True
    use_compile = False
    fullgraph = False

    do_profile = False
    profile_size = 1 << 22
    profile_warmup_steps = 1
    profile_active_steps = 3
    profile_trace_path = "logs/trace.json"

    sizes = [2**i for i in range(12, 27)]

    quantiles = [0.5]
    target_points_per_size = 1 << 27
    max_inner_repeats = 250
    cache_thrash_bytes = 64 << 20
    sync_repeats = 25

    if dims not in (2, 3):
        raise ValueError(f"dims must be 2 or 3, got {dims}")
    if direction not in {"encode", "decode"}:
        raise ValueError(f"direction must be 'encode' or 'decode', got {direction!r}")

    import numba as nb

    nb.set_num_threads(4)

    precache_compile_luts(device=device)

    cache = None
    if device.type in {"cuda", "hip"}:
        try:
            cache = runtime.driver.active.get_empty_cache_for_benchmark()  # type: ignore[attr-defined]
        except Exception:
            cache = None

    def run_op(inputs: tuple[torch.Tensor, ...], out: tuple[torch.Tensor, ...] | None):
        if direction == "encode":
            if dims == 2:
                x, y = inputs
                out_idx = None if out is None else out[0]
                return hilbert_encode_2d(
                    x,
                    y,
                    nbits=nbits,
                    out=out_idx,
                    cpu_backend=cpu_backend,
                    gpu_backend=gpu_backend,
                    cpu_parallel=True,
                )

            x, y, z = inputs
            out_idx = None if out is None else out[0]
            return hilbert_encode_3d(
                x,
                y,
                z,
                nbits=nbits,
                out=out_idx,
                cpu_backend=cpu_backend,
                gpu_backend=gpu_backend,
                cpu_parallel=True,
            )

        if dims == 2:
            index = inputs[0]
            out_x = None if out is None else out[0]
            out_y = None if out is None else out[1]
            return hilbert_decode_2d(
                index,
                nbits=nbits,
                out_x=out_x,
                out_y=out_y,
                cpu_backend=cpu_backend,
                gpu_backend=gpu_backend,
                cpu_parallel=True,
            )

        index = inputs[0]
        out_x = None if out is None else out[0]
        out_y = None if out is None else out[1]
        out_z = None if out is None else out[2]
        return hilbert_decode_3d(
            index,
            nbits=nbits,
            out_x=out_x,
            out_y=out_y,
            out_z=out_z,
            cpu_backend=cpu_backend,
            gpu_backend=gpu_backend,
            cpu_parallel=True,
        )

    def make_inputs(n: int) -> tuple[torch.Tensor, ...]:
        high = 1 << nbits
        if direction == "encode":
            if dims == 2:
                x = torch.empty((n,), dtype=dtype, device=device)
                y = torch.empty((n,), dtype=dtype, device=device)
                x.random_(0, high)
                y.random_(0, high)
                return (x, y)

            x = torch.empty((n,), dtype=dtype, device=device)
            y = torch.empty((n,), dtype=dtype, device=device)
            z = torch.empty((n,), dtype=dtype, device=device)
            x.random_(0, high)
            y.random_(0, high)
            z.random_(0, high)
            return (x, y, z)

        index_bits = dims * nbits
        if index_bits > 64:
            raise ValueError(
                f"decode index space exceeds 64 bits for uint64 output: {index_bits}"
            )

        index = torch.empty((n,), dtype=out_dtype, device=device)
        if index_bits <= 62:
            index.random_(0, 1 << index_bits)
        elif index_bits == 63:
            # random_ upper bounds are parsed as signed 64-bit; 2**63 overflows.
            index.copy_(
                torch.randint(
                    0, (1 << 63) - 1, (n,), dtype=torch.int64, device=device
                ).to(torch.uint64)
            )
        else:
            # Avoid overflow in random_ upper bound and avoid uint64 bitwise ops
            # on CUDA (e.g., lshift_cuda for UInt64 is not implemented).
            index.copy_(
                torch.randint(
                    0, (1 << 63) - 1, (n,), dtype=torch.int64, device=device
                ).to(torch.uint64)
            )
        return (index,)

    def make_out(n: int) -> tuple[torch.Tensor, ...] | None:
        if not use_out:
            return None

        if direction == "encode":
            return (torch.empty((n,), dtype=out_dtype, device=device),)

        if dims == 2:
            return (
                torch.empty((n,), dtype=dtype, device=device),
                torch.empty((n,), dtype=dtype, device=device),
            )

        return (
            torch.empty((n,), dtype=dtype, device=device),
            torch.empty((n,), dtype=dtype, device=device),
            torch.empty((n,), dtype=dtype, device=device),
        )

    inputs = make_inputs(1)
    out = make_out(1)

    if use_compile:
        try:
            run_op = torch.compile(run_op, fullgraph=fullgraph)  # type: ignore[attr-defined]
        except Exception as e:
            raise RuntimeError(
                "use_compile=True requested, but torch.compile failed"
            ) from e

    if do_profile:
        # Keep the trace small: profile a single size and a few steps.
        n = int(profile_size)
        inputs = make_inputs(n)
        out = make_out(n)

        # Warm-up outside profiler.
        _ = run_op(inputs, out)
        if use_compile:
            _ = run_op(inputs, out)

        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        schedule = torch.profiler.schedule(
            wait=0,
            warmup=int(profile_warmup_steps),
            active=int(profile_active_steps),
            repeat=1,
        )

        print(
            f"profiling: size={n}, warmup_steps={profile_warmup_steps}, "
            f"active_steps={profile_active_steps}, trace={profile_trace_path}"
        )

        with torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            record_shapes=False,
            with_stack=False,
            profile_memory=False,
        ) as prof:
            for _i in range(int(profile_warmup_steps) + int(profile_active_steps)):
                _ = run_op(inputs, out)
                prof.step()

        prof.export_chrome_trace(profile_trace_path)
        return

    print(
        f"device={device}, dims={dims}, direction={direction}, dtype={dtype}, "
        f"nbits={nbits}, use_out={use_out}, use_compile={use_compile}"
    )

    for n in sizes:
        inputs = make_inputs(n)
        out = make_out(n)

        thrash = None
        if cache is None:
            # Fallback for CPU (and any environment where Triton's cache flush
            # isn't available): touch a large buffer between runs.
            thrash = torch.empty(
                (cache_thrash_bytes,), dtype=torch.uint8, device=device
            )
            thrash.add_(1)

        # Warm-up (not timed)
        out0 = run_op(inputs, out)

        in_bytes = sum(t.element_size() for t in inputs)
        if isinstance(out0, tuple):
            out_bytes = sum(t.element_size() for t in out0)
        else:
            out_bytes = out0.element_size()
        bytes_per_point = in_bytes + out_bytes

        if use_compile:
            # Warm up compilation + runtime caches outside timing.
            _ = run_op(inputs, out)

        bench = triton.testing.do_bench(
            lambda: run_op(inputs, out),
            quantiles=quantiles,
        )
        if bench is None:
            dobench_ms = float("nan")
        elif isinstance(bench, (tuple, list)):
            dobench_ms = float(bench[0])
        else:
            dobench_ms = float(bench)

        def _rate_gb_s(t_ms: float) -> float:
            t_s = t_ms * 1e-3
            return (n * bytes_per_point) / t_s / 1e9

        def _rate_mpts_s(t_ms: float) -> float:
            t_s = t_ms * 1e-3
            return (n / t_s) / 1e6

        # Manual timing: cache-thrash before each timed call.
        # (Thrash is outside timing but we synchronize so it takes effect.)
        inner_repeats = max(1, min(max_inner_repeats, target_points_per_size // n))
        repeats = min(sync_repeats, inner_repeats)

        total_s = 0.0
        for _i in range(repeats):
            if cache is not None:
                runtime.driver.active.clear_cache(cache)  # type: ignore[attr-defined]
            else:
                assert thrash is not None
                thrash.add_(1)
            _sync_if_cuda(device)
            t0 = time.perf_counter()
            _ = run_op(inputs, out)
            _sync_if_cuda(device)
            t1 = time.perf_counter()
            total_s += t1 - t0
        sync_ms = (total_s / repeats) * 1e3

        print(
            f"n={n:>9,}  "
            f"do_bench={dobench_ms * 1e3:>8.2f}us  {_rate_mpts_s(dobench_ms):>8.2f} Mpts/s  {_rate_gb_s(dobench_ms):>8.2f} GB/s  "
            f"sync={sync_ms * 1e3:>8.2f}us  {_rate_mpts_s(sync_ms):>8.2f} Mpts/s  {_rate_gb_s(sync_ms):>8.2f} GB/s"
        )


if __name__ == "__main__":
    main()
