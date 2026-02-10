from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, cast

import numba as nb
import numpy as np

RateUnit = Literal["m", "k", "auto"]


def _load_module_from_file(file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from file: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module.__name__] = module
    spec.loader.exec_module(module)
    return module


def _load_object(spec: str) -> Any:
    """Load `module:object` or `path/to/file.py:object`."""
    if ":" not in spec:
        raise ValueError(
            "Implementation spec must be of the form 'module:object' or 'file.py:object'"
        )

    mod_part, obj_part = spec.split(":", 1)
    mod_part = mod_part.strip()
    obj_part = obj_part.strip()
    if not obj_part:
        raise ValueError(f"Missing object part in spec: {spec!r}")

    # file path
    mod_path = Path(mod_part)
    if mod_path.suffix == ".py" and mod_path.exists():
        module = _load_module_from_file(mod_path.resolve())
    else:
        module = importlib.import_module(mod_part)

    obj: Any = module
    for attr in obj_part.split("."):
        obj = getattr(obj, attr)
    return obj


def _resolve_impls(
    obj: Any,
    hilbert_implementation_type: type,
    *,
    nbits: int,
    config: Any,
) -> dict[str, Any]:
    """Resolve an object into one or more HilbertImplementation instances."""
    if isinstance(obj, hilbert_implementation_type):
        return {obj.name: obj}

    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(v, hilbert_implementation_type):
                out[str(k)] = v
            else:
                raise TypeError(
                    f"Dict values must be HilbertImplementation, got {type(v)!r} for key {k!r}"
                )
        return out

    if isinstance(obj, (list, tuple)):
        out: dict[str, Any] = {}
        for item in obj:
            out.update(
                _resolve_impls(
                    item,
                    hilbert_implementation_type,
                    nbits=nbits,
                    config=config,
                )
            )
        return out

    if callable(obj):
        # Allow factories that depend on `nbits` and/or `config`.
        try:
            sig = inspect.signature(obj)
            pos_params = [
                p
                for p in sig.parameters.values()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
        except (TypeError, ValueError):
            # Fall back to a few common call conventions.
            for args in ((), (nbits,), (config,), (nbits, config), (config, nbits)):
                try:
                    return _resolve_impls(
                        obj(*args),
                        hilbert_implementation_type,
                        nbits=nbits,
                        config=config,
                    )
                except TypeError:
                    continue
            raise

        argc = len(pos_params)
        if argc == 0:
            produced = obj()
        elif argc == 1:
            pname = pos_params[0].name
            if pname in {"nbits"}:
                produced = obj(nbits)
            elif pname in {"cfg", "config"}:
                produced = obj(config)
            else:
                # Try config first, then nbits.
                try:
                    produced = obj(config)
                except TypeError:
                    produced = obj(nbits)
        elif argc == 2:
            p0 = pos_params[0].name
            p1 = pos_params[1].name
            if p0 in {"nbits"} and p1 in {"cfg", "config"}:
                produced = obj(nbits, config)
            elif p0 in {"cfg", "config"} and p1 in {"nbits"}:
                produced = obj(config, nbits)
            else:
                produced = obj(nbits, config)
        else:
            raise TypeError(
                "Implementation factory must take 0, 1, or 2 positional args: "
                "(), (nbits|config), or (nbits, config)"
            )

        return _resolve_impls(
            produced,
            hilbert_implementation_type,
            nbits=nbits,
            config=config,
        )

    raise TypeError(
        "Implementation object must be HilbertImplementation, dict[str, HilbertImplementation], "
        "a list/tuple of those, or a callable returning one of those"
    )


def _make_pack_impl(nbits: int, ndim: int, hilbert_implementation_type: type) -> Any:
    """A simple reversible baseline (NOT Hilbert).

    - ndim=2: idx = (x<<nbits) | y
    - ndim=3: idx = (x<<(2*nbits)) | (y<<nbits) | z
    """

    ndim = int(ndim)
    if ndim not in (2, 3):
        raise ValueError("ndim must be 2 or 3")

    if ndim == 2:

        @nb.njit
        def encode2(xs: np.ndarray, ys: np.ndarray, out: np.ndarray) -> None:
            sh = np.uint64(nbits)
            for i in range(xs.shape[0]):
                out[i] = (np.uint64(xs[i]) << sh) | np.uint64(ys[i])

        @nb.njit
        def decode2(idx: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> None:
            sh = np.uint64(nbits)
            mask = np.uint64((1 << nbits) - 1)
            for i in range(idx.shape[0]):
                v = np.uint64(idx[i])
                xs[i] = np.uint64((v >> sh) & mask)
                ys[i] = np.uint64(v & mask)

        return hilbert_implementation_type(
            name=f"pack{nbits}", encode=encode2, decode=decode2
        )

    # ndim == 3

    @nb.njit
    def encode3(
        xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, out: np.ndarray
    ) -> None:
        sh = np.uint64(nbits)
        sh2 = np.uint64(2 * nbits)
        for i in range(xs.shape[0]):
            out[i] = (
                (np.uint64(xs[i]) << sh2) | (np.uint64(ys[i]) << sh) | np.uint64(zs[i])
            )

    @nb.njit
    def decode3(
        idx: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray
    ) -> None:
        sh = np.uint64(nbits)
        sh2 = np.uint64(2 * nbits)
        mask = np.uint64((1 << nbits) - 1)
        for i in range(idx.shape[0]):
            v = np.uint64(idx[i])
            xs[i] = np.uint64((v >> sh2) & mask)
            ys[i] = np.uint64((v >> sh) & mask)
            zs[i] = np.uint64(v & mask)

    return hilbert_implementation_type(
        name=f"pack3d{nbits}", encode=encode3, decode=decode3
    )


def main(argv: list[str] | None = None) -> int:
    # Make `scripts/hilbert` importable so `import hilbert_bench` (and plugin
    # modules that import it) works when this is executed as a script.
    this_dir = Path(__file__).resolve().parent
    if str(this_dir) not in sys.path:
        sys.path.insert(0, str(this_dir))

    from hilbert_bench import HilbertBench, HilbertBenchConfig, HilbertImplementation

    p = argparse.ArgumentParser(
        description=(
            "Standalone Hilbert bench runner.\n\n"
            "Pluggable implementations:\n"
            "  --impl mymodule:IMPLS\n"
            "  --impl path/to/file.py:get_impls\n\n"
            "Where the loaded object is either:\n"
            "  - HilbertImplementation\n"
            "  - dict[str, HilbertImplementation]\n"
            "  - list/tuple of HilbertImplementation\n"
            "  - callable returning any of the above\n"
        )
    )

    p.add_argument("--mode", choices=["grid", "arange", "random"], default="random")
    p.add_argument(
        "--ndim",
        type=int,
        choices=(2, 3),
        default=2,
        help="Number of dimensions (2 or 3)",
    )
    p.add_argument("--nbits", type=int, default=16)
    p.add_argument("--n", type=int, default=5_000_000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--threads", type=int, default=0)
    # Back-compat: --repeats used to exist; keep it as an alias for trials.
    p.add_argument(
        "--repeats",
        dest="trials",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of full-pass timing trials; reports median + stddev across trials",
    )
    p.add_argument(
        "--min-time",
        type=float,
        default=0.25,
        help="Minimum total seconds across all trials (0 disables)",
    )
    p.add_argument("--validate", action="store_true")
    p.add_argument("--validate-n", type=int, default=10_000)

    p.add_argument(
        "--impl",
        action="append",
        default=[],
        help="Implementation spec: module:object or file.py:object (repeatable)",
    )
    p.add_argument(
        "--impl-as",
        action="append",
        default=[],
        help="Alias an implementation spec: name=module:object (repeatable)",
    )
    p.add_argument(
        "--builtin",
        action="append",
        default=[],
        choices=["pack"],
        help="Add builtin implementation(s) (repeatable). 'pack' is a reversible baseline, not Hilbert.",
    )

    p.add_argument("--no-setup", action="store_true", help="Do not print setup")
    p.add_argument("--quiet", action="store_true", help="Do not print per-impl results")
    p.add_argument(
        "--rate-unit",
        choices=["m", "k", "auto"],
        default="m",
        help="Throughput unit for printed results: Mpts/s, Kpts/s, or auto",
    )
    p.add_argument(
        "--json",
        type=str,
        default="",
        help="Write results to a JSON file (includes config + per-impl results)",
    )

    args = p.parse_args(argv)

    cfg = HilbertBenchConfig(
        mode=args.mode,
        ndim=int(args.ndim),
        nbits=int(args.nbits),
        n=int(args.n),
        seed=int(args.seed),
        threads=int(args.threads),
        trials=int(args.trials) if args.trials is not None else 5,
        min_time_s=float(args.min_time),
        validate=bool(args.validate),
        validate_n=int(args.validate_n),
    )
    bench = HilbertBench(cfg)

    impls: dict[str, HilbertImplementation] = {}

    # builtins
    for b in args.builtin:
        if b == "pack":
            impl = _make_pack_impl(
                int(args.nbits), int(args.ndim), HilbertImplementation
            )
            impls[impl.name] = impl

    # --impl
    for spec in args.impl:
        obj = _load_object(spec)
        impls.update(
            _resolve_impls(
                obj,
                HilbertImplementation,
                nbits=int(cfg.nbits),
                config=cfg,
            )
        )

    # --impl-as
    for item in args.impl_as:
        if "=" not in item:
            raise SystemExit("--impl-as must be NAME=module:object")
        name, spec = item.split("=", 1)
        name = name.strip()
        spec = spec.strip()
        if not name:
            raise SystemExit("--impl-as must include a non-empty NAME")
        obj = _load_object(spec)
        resolved = _resolve_impls(
            obj,
            HilbertImplementation,
            nbits=int(cfg.nbits),
            config=cfg,
        )
        if len(resolved) != 1:
            raise SystemExit("--impl-as must resolve to exactly one implementation")
        impl = next(iter(resolved.values()))
        impls[name] = HilbertImplementation(
            name=name, encode=impl.encode, decode=impl.decode
        )

    if not impls:
        raise SystemExit(
            "No implementations provided. Use --impl and/or --builtin pack"
        )

    all_results = bench.run_many(
        impls,
        print_setup=not args.no_setup,
        print_results=not args.quiet,
        rate_unit=cast(RateUnit, args.rate_unit),
    )

    if args.json:
        out_path = Path(args.json)
        payload: dict[str, Any] = {
            "config": asdict(cfg),
            "numba_threads": nb.get_num_threads(),
            "results": {
                impl_name: {k: asdict(v) for k, v in res.items()}
                for impl_name, res in all_results.items()
            },
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
