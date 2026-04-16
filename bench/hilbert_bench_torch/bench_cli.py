import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import torch
import triton
from hilbert_skilling import decode as skilling_decode_torch
from hilbert_skilling import encode as skilling_encode_torch

from hilbertsfc.torch import (
    hilbert_decode_2d,
    hilbert_decode_3d,
    hilbert_encode_2d,
    hilbert_encode_3d,
    precache_compile_luts,
)

QUANTILES = [0.5]
OPS = ("encode", "decode")
ALL_PROVIDERS = (
    "skilling_eager",
    "hilbertsfc_torch_eager",
    "hilbertsfc_torch_compile",
    "hilbertsfc_triton",
    "skilling_triton_2d",
    "skilling_triton_3d",
)

_DTYPE_MAP: dict[str, torch.dtype] = {
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "uint16": torch.uint16,
    "uint32": torch.uint32,
    "uint64": torch.uint64,
}

_COMPILED_CACHE: dict[
    tuple[str, int, int, int, str, str, str], Callable[..., object]
] = {}
_PRECACHE_DONE: set[tuple[str, str]] = set()


@dataclass
class BenchRow:
    op: str
    dim: int
    nbits: int
    size: int
    provider: str
    available: bool
    error: str
    in_dtype: str
    out_dtypes: str
    bytes_per_point: int
    ms_p50: float
    ms_p20: float
    ms_p80: float
    mpts_p50: float
    mpts_p20: float
    mpts_p80: float
    gbps_p50: float
    gbps_p20: float
    gbps_p80: float


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _canonical_device(device: torch.device | str) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return dev


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _parse_dtype(name: str) -> torch.dtype:
    key = name.strip().lower()
    if key not in _DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype '{name}'. Supported: {sorted(_DTYPE_MAP.keys())}"
        )
    return _DTYPE_MAP[key]


def _safe_float(value: float) -> float | None:
    if math.isnan(value) or math.isinf(value):
        return None
    return float(value)


def _serialize_row(row: BenchRow) -> dict[str, object]:
    data = asdict(row)
    for key, value in list(data.items()):
        if isinstance(value, float):
            data[key] = _safe_float(value)
    return data


def _mpts(size: int, ms: float) -> float:
    if ms <= 0 or math.isnan(ms):
        return float("nan")
    return (size / (ms * 1e-3)) / 1e6


def _gbps(size: int, ms: float, bytes_per_point: int) -> float:
    if ms <= 0 or math.isnan(ms):
        return float("nan")
    return (size * bytes_per_point) / (ms * 1e-3) / 1e9


def _extract_tensors(result: object) -> list[torch.Tensor]:
    if isinstance(result, torch.Tensor):
        return [result]
    if isinstance(result, tuple | list):
        tensors = [t for t in result if isinstance(t, torch.Tensor)]
        if tensors:
            return tensors
    raise TypeError("provider did not return torch.Tensor outputs")


def _parse_bench_triplet(bench_result) -> tuple[float, float, float]:
    if bench_result is None:
        raise RuntimeError("triton.testing.do_bench returned None")
    if isinstance(bench_result, tuple | list):
        if len(bench_result) < 3:
            raise RuntimeError("triton.testing.do_bench returned malformed tuple")
        return (
            float(bench_result[0]),
            float(bench_result[1]),
            float(bench_result[2]),
        )
    ms = float(bench_result)
    return ms, ms, ms


def _make_row_unavailable(
    *,
    op: str,
    dim: int,
    nbits: int,
    size: int,
    provider: str,
    error: str,
    in_dtype: str,
) -> BenchRow:
    nan = float("nan")
    return BenchRow(
        op=op,
        dim=dim,
        nbits=nbits,
        size=size,
        provider=provider,
        available=False,
        error=error,
        in_dtype=in_dtype,
        out_dtypes="",
        bytes_per_point=0,
        ms_p50=nan,
        ms_p20=nan,
        ms_p80=nan,
        mpts_p50=nan,
        mpts_p20=nan,
        mpts_p80=nan,
        gbps_p50=nan,
        gbps_p20=nan,
        gbps_p80=nan,
    )


def _bench_provider(
    *,
    op: str,
    dim: int,
    nbits: int,
    size: int,
    provider: str,
    run_once: Callable[[], object],
    in_bytes_per_point: int,
    in_dtype: str,
    bench_warmup_calls: int,
) -> BenchRow:
    try:
        warm = run_once()
        outs = _extract_tensors(warm)

        for _ in range(max(0, int(bench_warmup_calls))):
            _ = run_once()

        if any(t.is_cuda for t in outs):
            torch.cuda.synchronize(outs[0].device)

        out_bytes = sum(int(t.element_size()) for t in outs)
        bytes_per_point = int(in_bytes_per_point + out_bytes)
        out_dtypes = ",".join(_dtype_name(t.dtype) for t in outs)

        bench = triton.testing.do_bench(run_once, quantiles=QUANTILES)
        ms_p50, ms_p20, ms_p80 = _parse_bench_triplet(bench)

        return BenchRow(
            op=op,
            dim=dim,
            nbits=nbits,
            size=size,
            provider=provider,
            available=True,
            error="",
            in_dtype=in_dtype,
            out_dtypes=out_dtypes,
            bytes_per_point=bytes_per_point,
            ms_p50=ms_p50,
            ms_p20=ms_p20,
            ms_p80=ms_p80,
            mpts_p50=_mpts(size, ms_p50),
            mpts_p20=_mpts(size, ms_p20),
            mpts_p80=_mpts(size, ms_p80),
            gbps_p50=_gbps(size, ms_p50, bytes_per_point),
            gbps_p20=_gbps(size, ms_p20, bytes_per_point),
            gbps_p80=_gbps(size, ms_p80, bytes_per_point),
        )
    except Exception as exc:
        return _make_row_unavailable(
            op=op,
            dim=dim,
            nbits=nbits,
            size=size,
            provider=provider,
            error=f"{type(exc).__name__}: {exc}",
            in_dtype=in_dtype,
        )


def _ensure_precache_compile_luts(
    op: str, dim: int, device: torch.device | str
) -> None:
    if op not in OPS:
        raise ValueError(f"Unsupported operation: {op}")
    op_name = f"hilbert_{op}_{dim}d"
    dev = _canonical_device(device)
    key = (str(dev), op_name)
    if key in _PRECACHE_DONE:
        return
    precache_compile_luts(device=dev, op=op_name)  # type: ignore[reportArgumentType]
    _PRECACHE_DONE.add(key)


def _provider_sequence(dim: int) -> list[str]:
    providers = [
        "skilling_eager",
        "hilbertsfc_torch_eager",
        "hilbertsfc_triton",
    ]
    if dim == 2:
        providers.append("skilling_triton_2d")
    else:
        providers.append("skilling_triton_3d")
    providers.append("hilbertsfc_torch_compile")
    return providers


def _try_load_skilling_triton_fn(op: str, dim: int):
    module_dir = str(Path(__file__).resolve().parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    symbol = f"{op}_{dim}d_triton"
    try:
        mod = __import__("hilbert_skilling_triton", fromlist=[symbol])
        return getattr(mod, symbol, None)
    except Exception:
        return None


def _build_coords(
    *,
    dim: int,
    size: int,
    nbits: int,
    device: str,
    seed: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    upper = 1 << int(nbits)
    generator_device = "cuda" if device.startswith("cuda") else "cpu"
    g = torch.Generator(device=generator_device)
    g.manual_seed(int(seed))

    coords = []
    for _ in range(dim):
        coord = torch.randint(
            0,
            upper,
            (size,),
            device=device,
            dtype=torch.int64,
            generator=g,
        ).to(dtype)
        coords.append(coord)

    return tuple(coords)


def _build_index(
    *,
    dim: int,
    size: int,
    nbits: int,
    device: str,
    seed: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    total_bits = int(dim * nbits)
    generator_device = "cuda" if device.startswith("cuda") else "cpu"
    g = torch.Generator(device=generator_device)
    g.manual_seed(int(seed))

    if total_bits <= 62:
        high = int(1 << total_bits)
    else:
        # int64 torch.randint cannot use high >= 2**63.
        high = (1 << 63) - 1

    base_i64 = torch.randint(
        0,
        int(max(2, high)),
        (size,),
        device=device,
        dtype=torch.int64,
        generator=g,
    )

    if total_bits < 63:
        base_i64 = base_i64 & int((1 << total_bits) - 1)

    return base_i64.to(dtype)


def _get_compiled_hsfc_torch(
    *,
    op: str,
    dim: int,
    nbits: int,
    size: int,
    device: torch.device | str,
    coord_dtype: torch.dtype,
    index_dtype: torch.dtype,
):
    if not hasattr(torch, "compile"):
        return None

    dev = _canonical_device(device)
    key = (
        op,
        int(dim),
        int(nbits),
        int(size),
        str(dev),
        _dtype_name(coord_dtype),
        _dtype_name(index_dtype),
    )
    fn = _COMPILED_CACHE.get(key)
    if fn is not None:
        return fn

    try:
        _ensure_precache_compile_luts(op, dim, dev)

        if op == "encode":
            if dim == 2:

                def _f(x: torch.Tensor, y: torch.Tensor):  # type: ignore[reportRedeclaration]
                    out = torch.empty_like(x, dtype=index_dtype)
                    return hilbert_encode_2d(
                        x,
                        y,
                        nbits=nbits,
                        out=out,
                        gpu_backend="torch",
                    )

            else:

                def _f(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):  # type: ignore[reportRedeclaration]
                    out = torch.empty_like(x, dtype=index_dtype)
                    return hilbert_encode_3d(
                        x,
                        y,
                        z,
                        nbits=nbits,
                        out=out,
                        gpu_backend="torch",
                    )

        else:
            if dim == 2:

                def _f(index: torch.Tensor):  # type: ignore[reportRedeclaration]
                    out_x = torch.empty_like(index, dtype=coord_dtype)
                    out_y = torch.empty_like(index, dtype=coord_dtype)
                    return hilbert_decode_2d(
                        index,
                        nbits=nbits,
                        out_x=out_x,
                        out_y=out_y,
                        gpu_backend="torch",
                    )

            else:

                def _f(index: torch.Tensor):
                    out_x = torch.empty_like(index, dtype=coord_dtype)
                    out_y = torch.empty_like(index, dtype=coord_dtype)
                    out_z = torch.empty_like(index, dtype=coord_dtype)
                    return hilbert_decode_3d(
                        index,
                        nbits=nbits,
                        out_x=out_x,
                        out_y=out_y,
                        out_z=out_z,
                        gpu_backend="torch",
                    )

        fn = torch.compile(_f)
    except Exception:
        return None

    _COMPILED_CACHE[key] = fn
    return fn


def _make_skilling_eager_run_once(
    *,
    op: str,
    dim: int,
    nbits: int,
    locs: torch.Tensor | None,
    index: torch.Tensor | None,
    coord_dtype: torch.dtype,
    index_dtype: torch.dtype,
) -> Callable[[], object]:
    if op == "encode":
        if locs is None:
            raise ValueError("encode requires locs")

        def _run_once() -> object:
            return skilling_encode_torch(locs, num_dims=dim, num_bits=nbits).to(
                index_dtype
            )

        return _run_once

    if index is None:
        raise ValueError("decode requires index")

    def _run_once() -> object:
        decoded = skilling_decode_torch(index, num_dims=dim, num_bits=nbits)
        if dim == 2:
            return decoded[:, 0].to(coord_dtype), decoded[:, 1].to(coord_dtype)
        return (
            decoded[:, 0].to(coord_dtype),
            decoded[:, 1].to(coord_dtype),
            decoded[:, 2].to(coord_dtype),
        )

    return _run_once


def _make_hilbertsfc_run_once(
    *,
    op: str,
    dim: int,
    nbits: int,
    coords: tuple[torch.Tensor, ...] | None,
    index: torch.Tensor | None,
    coord_dtype: torch.dtype,
    index_dtype: torch.dtype,
    gpu_backend: Literal["torch", "triton"],
) -> Callable[[], object]:
    if op == "encode":
        if coords is None:
            raise ValueError("encode requires coords")

        if dim == 2:

            def _run_once() -> object:
                out = torch.empty_like(coords[0], dtype=index_dtype)
                return hilbert_encode_2d(
                    coords[0],
                    coords[1],
                    nbits=nbits,
                    out=out,
                    gpu_backend=gpu_backend,
                )

            return _run_once

        def _run_once() -> object:
            out = torch.empty_like(coords[0], dtype=index_dtype)
            return hilbert_encode_3d(
                coords[0],
                coords[1],
                coords[2],
                nbits=nbits,
                out=out,
                gpu_backend=gpu_backend,
            )

        return _run_once

    if index is None:
        raise ValueError("decode requires index")

    if dim == 2:

        def _run_once() -> object:
            out_x = torch.empty_like(index, dtype=coord_dtype)
            out_y = torch.empty_like(index, dtype=coord_dtype)
            return hilbert_decode_2d(
                index,
                nbits=nbits,
                out_x=out_x,
                out_y=out_y,
                gpu_backend=gpu_backend,
            )

        return _run_once

    def _run_once() -> object:
        out_x = torch.empty_like(index, dtype=coord_dtype)
        out_y = torch.empty_like(index, dtype=coord_dtype)
        out_z = torch.empty_like(index, dtype=coord_dtype)
        return hilbert_decode_3d(
            index,
            nbits=nbits,
            out_x=out_x,
            out_y=out_y,
            out_z=out_z,
            gpu_backend=gpu_backend,
        )

    return _run_once


def _make_compiled_run_once(
    *,
    op: str,
    dim: int,
    compiled: Callable[..., object],
    coords: tuple[torch.Tensor, ...] | None,
    index: torch.Tensor | None,
) -> Callable[[], object]:
    if op == "encode":
        if coords is None:
            raise ValueError("encode requires coords")
        if dim == 2:
            fn2 = cast(Callable[[torch.Tensor, torch.Tensor], object], compiled)

            def _run_once() -> object:
                return fn2(coords[0], coords[1])

            return _run_once

        fn3 = cast(
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object],
            compiled,
        )

        def _run_once() -> object:
            return fn3(coords[0], coords[1], coords[2])

        return _run_once

    if index is None:
        raise ValueError("decode requires index")
    fn1 = cast(Callable[[torch.Tensor], object], compiled)

    def _run_once() -> object:
        return fn1(index)

    return _run_once


def _make_skilling_triton_run_once(
    *,
    op: str,
    dim: int,
    nbits: int,
    skilling_fn: Callable[..., object],
    coords: tuple[torch.Tensor, ...] | None,
    index: torch.Tensor | None,
    coord_dtype: torch.dtype,
    index_dtype: torch.dtype,
) -> Callable[[], object]:
    if op == "encode":
        if coords is None:
            raise ValueError("encode requires coords")
        if dim == 2:

            def _run_once() -> object:
                return skilling_fn(
                    coords[0],
                    coords[1],
                    num_bits=nbits,
                    out_dtype=index_dtype,
                )

            return _run_once

        def _run_once() -> object:
            return skilling_fn(
                coords[0],
                coords[1],
                coords[2],
                num_bits=nbits,
                out_dtype=index_dtype,
            )

        return _run_once

    if index is None:
        raise ValueError("decode requires index")

    def _run_once() -> object:
        return skilling_fn(index, num_bits=nbits, out_dtype=coord_dtype)

    return _run_once


def _run_single_provider_internal(
    *,
    op: str,
    provider: str,
    dim: int,
    size: int,
    nbits_2d: int,
    nbits_3d: int,
    device: str,
    seed: int,
    coord_dtype_2d: torch.dtype,
    coord_dtype_3d: torch.dtype,
    index_dtype_2d: torch.dtype,
    index_dtype_3d: torch.dtype,
    include_skilling_triton: bool,
    bench_warmup_calls: int,
) -> BenchRow:
    nbits = nbits_2d if dim == 2 else nbits_3d
    if dim * nbits > 64:
        raise ValueError(f"dim*nbits must be <= 64, got dim={dim}, nbits={nbits}")

    coord_dtype = coord_dtype_2d if dim == 2 else coord_dtype_3d
    index_dtype = index_dtype_2d if dim == 2 else index_dtype_3d

    local_seed = int(seed + dim * 1_000_000 + size)

    if op == "encode":
        coords = _build_coords(
            dim=dim,
            size=size,
            nbits=nbits,
            device=device,
            seed=local_seed,
            dtype=coord_dtype,
        )
        locs = torch.stack([c.to(torch.int64) for c in coords], dim=1)
        in_bytes_per_point = dim * int(coords[0].element_size())
        in_dtype = _dtype_name(coord_dtype)
        index = None
    else:
        index = _build_index(
            dim=dim,
            size=size,
            nbits=nbits,
            device=device,
            seed=local_seed,
            dtype=index_dtype,
        )
        in_bytes_per_point = int(index.element_size())
        in_dtype = _dtype_name(index_dtype)
        coords = None
        locs = None

    run_once: Callable[[], object]
    if provider == "skilling_eager":
        run_once = _make_skilling_eager_run_once(
            op=op,
            dim=dim,
            nbits=nbits,
            locs=locs,
            index=index,
            coord_dtype=coord_dtype,
            index_dtype=index_dtype,
        )

    elif provider == "hilbertsfc_torch_eager":
        run_once = _make_hilbertsfc_run_once(
            op=op,
            dim=dim,
            nbits=nbits,
            coords=coords,
            index=index,
            coord_dtype=coord_dtype,
            index_dtype=index_dtype,
            gpu_backend="torch",
        )

    elif provider == "hilbertsfc_torch_compile":
        compiled = _get_compiled_hsfc_torch(
            op=op,
            dim=dim,
            nbits=nbits,
            size=size,
            device=device,
            coord_dtype=coord_dtype,
            index_dtype=index_dtype,
        )
        if compiled is None:
            return _make_row_unavailable(
                op=op,
                dim=dim,
                nbits=nbits,
                size=size,
                provider=provider,
                error="torch.compile unavailable or failed",
                in_dtype=in_dtype,
            )
        run_once = _make_compiled_run_once(
            op=op,
            dim=dim,
            compiled=compiled,
            coords=coords,
            index=index,
        )

    elif provider == "hilbertsfc_triton":
        run_once = _make_hilbertsfc_run_once(
            op=op,
            dim=dim,
            nbits=nbits,
            coords=coords,
            index=index,
            coord_dtype=coord_dtype,
            index_dtype=index_dtype,
            gpu_backend="triton",
        )

    elif provider in ("skilling_triton_2d", "skilling_triton_3d"):
        if (provider.endswith("2d") and dim != 2) or (
            provider.endswith("3d") and dim != 3
        ):
            return _make_row_unavailable(
                op=op,
                dim=dim,
                nbits=nbits,
                size=size,
                provider=provider,
                error=f"provider only valid for dim={2 if provider.endswith('2d') else 3}",
                in_dtype=in_dtype,
            )
        if not include_skilling_triton:
            return _make_row_unavailable(
                op=op,
                dim=dim,
                nbits=nbits,
                size=size,
                provider=provider,
                error="optional kernel disabled",
                in_dtype=in_dtype,
            )

        skilling_fn = _try_load_skilling_triton_fn(op=op, dim=dim)
        if skilling_fn is None:
            return _make_row_unavailable(
                op=op,
                dim=dim,
                nbits=nbits,
                size=size,
                provider=provider,
                error="optional kernel unavailable",
                in_dtype=in_dtype,
            )
        run_once = _make_skilling_triton_run_once(
            op=op,
            dim=dim,
            nbits=nbits,
            skilling_fn=skilling_fn,
            coords=coords,
            index=index,
            coord_dtype=coord_dtype,
            index_dtype=index_dtype,
        )

    else:
        return _make_row_unavailable(
            op=op,
            dim=dim,
            nbits=nbits,
            size=size,
            provider=provider,
            error=f"unknown provider: {provider}",
            in_dtype=in_dtype,
        )

    return _bench_provider(
        op=op,
        dim=dim,
        nbits=nbits,
        size=size,
        provider=provider,
        run_once=run_once,
        in_bytes_per_point=in_bytes_per_point,
        in_dtype=in_dtype,
        bench_warmup_calls=bench_warmup_calls,
    )


def _isolated_input_dtype_name(
    *,
    op: str,
    dim: int,
    coord_dtype_2d: torch.dtype,
    coord_dtype_3d: torch.dtype,
    index_dtype_2d: torch.dtype,
    index_dtype_3d: torch.dtype,
) -> str:
    if op == "encode":
        return _dtype_name(coord_dtype_2d if dim == 2 else coord_dtype_3d)
    return _dtype_name(index_dtype_2d if dim == 2 else index_dtype_3d)


def _build_isolated_cmd(
    *,
    script_path: Path,
    op: str,
    provider: str,
    dim: int,
    size: int,
    row_out_path: Path,
    nbits_2d: int,
    nbits_3d: int,
    device: str,
    seed: int,
    coord_dtype_2d: torch.dtype,
    coord_dtype_3d: torch.dtype,
    index_dtype_2d: torch.dtype,
    index_dtype_3d: torch.dtype,
    include_skilling_triton: bool,
    bench_warmup_calls: int,
) -> list[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--internal-op",
        op,
        "--internal-single-provider",
        provider,
        "--internal-dim",
        str(dim),
        "--internal-size",
        str(size),
        "--internal-row-out",
        str(row_out_path),
        "--nbits-2d",
        str(nbits_2d),
        "--nbits-3d",
        str(nbits_3d),
        "--device",
        str(device),
        "--seed",
        str(seed),
        "--2d-coord-dtype",
        _dtype_name(coord_dtype_2d),
        "--3d-coord-dtype",
        _dtype_name(coord_dtype_3d),
        "--2d-index-dtype",
        _dtype_name(index_dtype_2d),
        "--3d-index-dtype",
        _dtype_name(index_dtype_3d),
        "--bench-warmup-calls",
        str(bench_warmup_calls),
    ]
    if not include_skilling_triton:
        cmd.append("--skip-skilling-triton")
    return cmd


def _run_isolated_provider(
    *,
    script_path: Path,
    op: str,
    provider: str,
    dim: int,
    size: int,
    nbits: int,
    nbits_2d: int,
    nbits_3d: int,
    device: str,
    seed: int,
    coord_dtype_2d: torch.dtype,
    coord_dtype_3d: torch.dtype,
    index_dtype_2d: torch.dtype,
    index_dtype_3d: torch.dtype,
    include_skilling_triton: bool,
    bench_warmup_calls: int,
) -> BenchRow:
    in_dtype = _isolated_input_dtype_name(
        op=op,
        dim=dim,
        coord_dtype_2d=coord_dtype_2d,
        coord_dtype_3d=coord_dtype_3d,
        index_dtype_2d=index_dtype_2d,
        index_dtype_3d=index_dtype_3d,
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        row_out_path = Path(tmp.name)

    try:
        cmd = _build_isolated_cmd(
            script_path=script_path,
            op=op,
            provider=provider,
            dim=dim,
            size=size,
            row_out_path=row_out_path,
            nbits_2d=nbits_2d,
            nbits_3d=nbits_3d,
            device=device,
            seed=seed,
            coord_dtype_2d=coord_dtype_2d,
            coord_dtype_3d=coord_dtype_3d,
            index_dtype_2d=index_dtype_2d,
            index_dtype_3d=index_dtype_3d,
            include_skilling_triton=include_skilling_triton,
            bench_warmup_calls=bench_warmup_calls,
        )
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)

        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip() or "child failed"
            return _make_row_unavailable(
                op=op,
                dim=dim,
                nbits=nbits,
                size=size,
                provider=provider,
                error=f"isolated provider run failed: {err[-240:]}",
                in_dtype=in_dtype,
            )

        payload = json.loads(row_out_path.read_text(encoding="utf-8"))
        return BenchRow(**payload)
    finally:
        row_out_path.unlink(missing_ok=True)


def _run_isolated(
    *,
    ops: list[str],
    dims: list[int],
    nbits_2d: int,
    nbits_3d: int,
    min_exp: int,
    max_exp: int,
    device: str,
    seed: int,
    coord_dtype_2d: torch.dtype,
    coord_dtype_3d: torch.dtype,
    index_dtype_2d: torch.dtype,
    index_dtype_3d: torch.dtype,
    include_skilling_triton: bool,
    bench_warmup_calls: int,
) -> list[BenchRow]:
    rows: list[BenchRow] = []
    script_path = Path(__file__).resolve()
    sizes = [2**i for i in range(min_exp, max_exp + 1)]

    for op in ops:
        for dim in dims:
            nbits = nbits_2d if dim == 2 else nbits_3d
            if dim * nbits > 64:
                raise ValueError(
                    f"dim*nbits must be <= 64, got dim={dim}, nbits={nbits}"
                )

            providers = _provider_sequence(dim)
            for size in sizes:
                for provider in providers:
                    rows.append(
                        _run_isolated_provider(
                            script_path=script_path,
                            op=op,
                            provider=provider,
                            dim=dim,
                            size=size,
                            nbits=nbits,
                            nbits_2d=nbits_2d,
                            nbits_3d=nbits_3d,
                            device=device,
                            seed=seed,
                            coord_dtype_2d=coord_dtype_2d,
                            coord_dtype_3d=coord_dtype_3d,
                            index_dtype_2d=index_dtype_2d,
                            index_dtype_3d=index_dtype_3d,
                            include_skilling_triton=include_skilling_triton,
                            bench_warmup_calls=bench_warmup_calls,
                        )
                    )

                print(
                    f"bench done: op={op} dim={dim} nbits={nbits} size={size:,} [isolated]"
                )

    return rows


def _write_csv(rows: list[BenchRow], out_path: Path) -> None:
    fieldnames = (
        list(asdict(rows[0]).keys()) if rows else list(BenchRow.__annotations__)
    )
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_json(rows: list[BenchRow], out_path: Path, meta: dict[str, object]) -> None:
    payload = {"meta": meta, "rows": [_serialize_row(r) for r in rows]}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_summary(rows: list[BenchRow]) -> None:
    print("\nSummary (p50):")
    print(
        "op      dim  provider                   size        mpts/s      gb/s        ms      status"
    )
    for row in rows:
        status = "ok" if row.available else "skip"
        mpts = "nan" if math.isnan(row.mpts_p50) else f"{row.mpts_p50:9.2f}"
        gbps = "nan" if math.isnan(row.gbps_p50) else f"{row.gbps_p50:9.2f}"
        ms = "nan" if math.isnan(row.ms_p50) else f"{row.ms_p50:8.3f}"
        print(
            f"{row.op:<7} {row.dim:>3}  {row.provider:<24} {row.size:>8,}  {mpts:>9}  {gbps:>9}  {ms:>8}  {status}"
        )


def _parse_ops(args: argparse.Namespace) -> list[str]:
    ops: set[str] = set()

    for op in args.op:
        if op == "both":
            ops.update(OPS)
        else:
            ops.add(op)

    if args.encode:
        ops.add("encode")
    if args.decode:
        ops.add("decode")

    if not ops:
        ops.update(OPS)

    return [op for op in OPS if op in ops]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark torch-based Hilbert encode/decode with isolated per-provider execution. "
            "Compares Skilling vs HilbertSFC (torch backend, triton backend, torch.compile)."
        )
    )
    p.add_argument(
        "--op", nargs="+", choices=["encode", "decode", "both"], default=["both"]
    )
    p.add_argument("--encode", action="store_true", help="Include encode op.")
    p.add_argument("--decode", action="store_true", help="Include decode op.")
    p.add_argument("--dims", type=int, nargs="+", default=[2, 3])
    p.add_argument("--nbits-2d", type=int, default=16)
    p.add_argument("--nbits-3d", type=int, default=16)
    p.add_argument(
        "--2d-coord-dtype", dest="coord_dtype_2d", type=str, default="uint64"
    )
    p.add_argument(
        "--3d-coord-dtype", dest="coord_dtype_3d", type=str, default="uint64"
    )
    p.add_argument(
        "--2d-index-dtype", dest="index_dtype_2d", type=str, default="uint64"
    )
    p.add_argument(
        "--3d-index-dtype", dest="index_dtype_3d", type=str, default="uint64"
    )
    p.add_argument("--min-exp", type=int, default=12)
    p.add_argument("--max-exp", type=int, default=21)
    p.add_argument("--device", type=str, default=_default_device())
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-dir", type=str, default=str(Path("bench/hilbert_bench_torch/results"))
    )
    p.add_argument(
        "--skip-skilling-triton",
        action="store_true",
        help="Disable optional Skilling Triton kernels from scripts/bench.",
    )
    p.add_argument("--bench-warmup-calls", type=int, default=2)

    p.add_argument(
        "--internal-op",
        type=str,
        choices=list(OPS),
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--internal-single-provider",
        type=str,
        choices=list(ALL_PROVIDERS),
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--internal-dim", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--internal-size", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--internal-row-out", type=str, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    coord_dtype_2d = _parse_dtype(args.coord_dtype_2d)
    coord_dtype_3d = _parse_dtype(args.coord_dtype_3d)
    index_dtype_2d = _parse_dtype(args.index_dtype_2d)
    index_dtype_3d = _parse_dtype(args.index_dtype_3d)

    if args.internal_single_provider is not None:
        if args.internal_op is None:
            raise ValueError("internal mode requires --internal-op")
        if (
            args.internal_dim is None
            or args.internal_size is None
            or not args.internal_row_out
        ):
            raise ValueError("internal mode requires dim, size, and row-out")

        row = _run_single_provider_internal(
            op=str(args.internal_op),
            provider=str(args.internal_single_provider),
            dim=int(args.internal_dim),
            size=int(args.internal_size),
            nbits_2d=int(args.nbits_2d),
            nbits_3d=int(args.nbits_3d),
            device=str(args.device),
            seed=int(args.seed),
            coord_dtype_2d=coord_dtype_2d,
            coord_dtype_3d=coord_dtype_3d,
            index_dtype_2d=index_dtype_2d,
            index_dtype_3d=index_dtype_3d,
            include_skilling_triton=not bool(args.skip_skilling_triton),
            bench_warmup_calls=int(args.bench_warmup_calls),
        )
        Path(args.internal_row_out).write_text(
            json.dumps(asdict(row)), encoding="utf-8"
        )
        return 0

    ops = _parse_ops(args)
    dims = [int(d) for d in args.dims]
    if any(d not in (2, 3) for d in dims):
        raise ValueError("--dims only supports 2 and 3")
    if args.max_exp < args.min_exp:
        raise ValueError("--max-exp must be >= --min-exp")
    if int(args.bench_warmup_calls) < 0:
        raise ValueError("--bench-warmup-calls must be >= 0")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _run_isolated(
        ops=ops,
        dims=dims,
        nbits_2d=int(args.nbits_2d),
        nbits_3d=int(args.nbits_3d),
        min_exp=int(args.min_exp),
        max_exp=int(args.max_exp),
        device=str(args.device),
        seed=int(args.seed),
        coord_dtype_2d=coord_dtype_2d,
        coord_dtype_3d=coord_dtype_3d,
        index_dtype_2d=index_dtype_2d,
        index_dtype_3d=index_dtype_3d,
        include_skilling_triton=not bool(args.skip_skilling_triton),
        bench_warmup_calls=int(args.bench_warmup_calls),
    )

    csv_path = out_dir / "bench_results.csv"
    json_path = out_dir / "bench_results.json"
    _write_csv(rows, csv_path)
    _write_json(
        rows,
        json_path,
        meta={
            "ops": ops,
            "device": str(args.device),
            "dims": dims,
            "nbits_2d": int(args.nbits_2d),
            "nbits_3d": int(args.nbits_3d),
            "coord_dtype_2d": _dtype_name(coord_dtype_2d),
            "coord_dtype_3d": _dtype_name(coord_dtype_3d),
            "index_dtype_2d": _dtype_name(index_dtype_2d),
            "index_dtype_3d": _dtype_name(index_dtype_3d),
            "min_exp": int(args.min_exp),
            "max_exp": int(args.max_exp),
            "quantiles": QUANTILES,
            "bench_warmup_calls": int(args.bench_warmup_calls),
            "isolation": "always_subprocess",
        },
    )

    _print_summary(rows)
    print(f"\nWrote CSV:  {csv_path}")
    print(f"Wrote JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
