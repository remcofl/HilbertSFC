import argparse
import importlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, cast

import numba as nb

RateUnit = Literal["m", "k", "auto"]

_BUILTIN_IMPL_SPECS: dict[int, dict[str, str]] = {
    2: {
        "hilbertsfc": "hilbert_impls_2d:HILBERTSFC_2D",
        "hilbertsfc-morton": "hilbert_impls_2d:HILBERTSFC_MORTON_2D",
        "hilbert-bytes": "hilbert_impls_2d:HILBERT_BYTES_2D",
        "numpy-hilbert-curve": "hilbert_impls_2d:NUMPY_HILBERT_CURVE_2D",
        "hilbertcurve": "hilbert_impls_2d:HILBERTCURVE_2D",
    },
    3: {
        "hilbertsfc": "hilbert_impls_3d:HILBERTSFC_3D",
        "hilbertsfc-morton": "hilbert_impls_3d:HILBERTSFC_MORTON_3D",
        "hilbert-bytes": "hilbert_impls_3d:HILBERT_BYTES_3D",
        "numpy-hilbert-curve": "hilbert_impls_3d:NUMPY_HILBERT_CURVE_3D",
        "hilbertcurve": "hilbert_impls_3d:HILBERTCURVE_3D",
    },
}


def _builtin_impl_names() -> list[str]:
    names: set[str] = set()
    for specs in _BUILTIN_IMPL_SPECS.values():
        names.update(specs.keys())
    return sorted(names)


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
            out.update(_resolve_impls(item, hilbert_implementation_type))
        return out

    raise TypeError(
        "Implementation object must be HilbertImplementation, dict[str, HilbertImplementation], "
        "or a list/tuple of those"
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
            "  --impl-name hilbertsfc\n"
            "  --impl mymodule:IMPLS\n"
            "  --impl path/to/file.py:IMPLS\n\n"
            "Where the loaded object is either:\n"
            "  - HilbertImplementation\n"
            "  - dict[str, HilbertImplementation]\n"
            "  - list/tuple of HilbertImplementation\n"
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
    p.add_argument("--threads", type=int, default=1)
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
        "--impl-name",
        action="append",
        default=[],
        choices=_builtin_impl_names(),
        help=(
            "Builtin implementation name (repeatable). Options: "
            + ", ".join(_builtin_impl_names())
        ),
    )
    p.add_argument(
        "--list-impls",
        action="store_true",
        help="List builtin implementation names for the selected ndim and exit",
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

    if args.list_impls:
        ndim = int(args.ndim)
        specs = _BUILTIN_IMPL_SPECS.get(ndim, {})
        print(f"builtin implementations for ndim={ndim}:")
        for name in sorted(specs.keys()):
            print(f"- {name}")
        return 0

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

    # --impl-name
    for name in args.impl_name:
        specs = _BUILTIN_IMPL_SPECS[int(cfg.ndim)]
        spec = specs.get(name)
        if not spec:
            raise SystemExit(
                f"Builtin implementation {name!r} is not available for ndim={int(cfg.ndim)}"
            )
        obj = _load_object(spec)
        resolved = _resolve_impls(obj, HilbertImplementation)
        if len(resolved) != 1:
            raise SystemExit(
                f"Builtin implementation {name!r} must resolve to exactly one implementation"
            )
        impls[name] = next(iter(resolved.values()))

    # --impl
    for spec in args.impl:
        obj = _load_object(spec)
        impls.update(_resolve_impls(obj, HilbertImplementation))

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
        resolved = _resolve_impls(obj, HilbertImplementation)
        if len(resolved) != 1:
            raise SystemExit("--impl-as must resolve to exactly one implementation")
        impl = next(iter(resolved.values()))
        impls[name] = HilbertImplementation(
            name=name, encode=impl.encode, decode=impl.decode
        )

    if not impls:
        raise SystemExit(
            "No implementations provided. Use --impl-name, --impl, and/or --impl-as. "
            "Use --list-impls to see builtin names for the selected --ndim."
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
