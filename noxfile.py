import os
import shutil
from collections.abc import Callable

import nox

# Keep CI output readable.
nox.options.reuse_existing_virtualenvs = False
nox.options.stop_on_first_error = False
nox.options.default_venv_backend = "uv"

PYTHON_VERSIONS: tuple[str, ...] = ("3.12", "3.13", "3.14")
STRICT_OPTIONAL_TESTS = os.environ.get("HILBERTSFC_STRICT_OPTIONAL_TESTS") == "1"


def _install(
    session: nox.Session,
    *,
    groups: list[str] | None = None,
    resolution: str | None = None,
    project: bool = False,
) -> None:
    install_args: list[str] = []
    if resolution is not None:
        install_args.extend(["--resolution", resolution])
    if project:
        install_args.extend(["-e", "."])
    if groups is not None:
        for group in groups:
            install_args.extend(["--group", group])
    session.install(*install_args)


def _show_versions(session: nox.Session, *, torch: bool = False) -> None:
    imports = "import numpy, numba"
    output = "print('numpy', numpy.__version__); print('numba', numba.__version__)"
    if torch:
        imports += ", torch"
        output += "; print('torch', torch.__version__)"
    session.run(
        "python",
        "-c",
        f"{imports}; {output}",
    )


BACKENDS: dict[str, Callable[[], bool]] = {
    "cuda": lambda: shutil.which("nvidia-smi") is not None,
    "rocm": lambda: shutil.which("rocminfo") is not None or os.path.exists("/opt/rocm"),
}

GPU_BACKEND_VARIANTS = [
    nox.param("cuda", "torch-cu130", id="cu130"),
    nox.param("rocm", "torch-rocm", id="rocm"),
]


def _skip_or_error(session: nox.Session, message: str) -> None:
    if STRICT_OPTIONAL_TESTS:
        session.error(message)
    session.skip(message)


def _skip_if_backend_unavailable(session: nox.Session, backend: str) -> None:
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported GPU backend: {backend}")
    if not BACKENDS[backend]():
        _skip_or_error(session, f"Backend {backend} not available")


def _skip_if_torch_compile_cpu_unavailable(session: nox.Session) -> None:
    compiler_names = (
        ("cl.exe", "clang-cl.exe", "icx-cl.exe")
        if os.name == "nt"
        else ("c++", "g++", "clang++")
    )
    if any(shutil.which(compiler) is not None for compiler in compiler_names):
        return

    if os.name == "nt":
        _skip_or_error(
            session,
            "torch.compile CPU tests require an active C++ compiler; "
            "run Nox from a Visual Studio Developer shell",
        )
    _skip_or_error(
        session,
        "torch.compile CPU tests require a C++ compiler; "
        f"install one of {', '.join(compiler_names)}",
    )


@nox.session(venv_backend="none")
def lint(session: nox.Session) -> None:
    """Run Ruff (lint + format check)."""
    session.run("uvx", "ruff", "check", "src", external=True)
    session.run("uvx", "ruff", "format", "--check", "src", external=True)


@nox.session(python=PYTHON_VERSIONS)
def typecheck(session: nox.Session) -> None:
    """Run type check."""
    _install(session, project=True, groups=["typecheck", "torch-cpu"])
    session.run("pyright", "src")
    # Faster alternatives, but still gaps in type resolution:
    # session.run("pyrefly", "check", "src")
    # session.run("ty", "check", "src")


@nox.session(python=PYTHON_VERSIONS)
def test(session: nox.Session) -> None:
    """Run the core unit tests against the default dependency resolver result."""
    _install(session, project=True, groups=["test"])
    _show_versions(session)
    session.run("pytest", "-q", "-n", "auto", "-m", "not torch")


@nox.session(python="3.12")
def test_min(session: nox.Session) -> None:
    """Run core unit tests with minimum supported numpy/numba (Python 3.12 only)."""
    _install(session, project=True, resolution="lowest-direct")
    _install(session, groups=["test"])
    _show_versions(session)
    session.run("pytest", "-q", "-n", "auto", "-m", "not torch")


@nox.session(python=PYTHON_VERSIONS)
def test_torch_cpu(session: nox.Session) -> None:
    """Run CPU-only torch frontend tests for regular CI."""

    _install(session, project=True, groups=["test", "torch-cpu"])

    session.run("pytest", "-q", "-m", "torch and not compile and not gpu")


@nox.session(python="3.12")
def test_torch_cpu_min(session: nox.Session) -> None:
    """Run CPU-only torch frontend tests with minimum deps (Python 3.12 only)."""

    _install(
        session,
        project=True,
        groups=["torch-cpu"],
        resolution="lowest-direct",
    )
    _install(session, groups=["test"])
    _show_versions(session, torch=True)

    session.run("pytest", "-q", "-m", "torch and not compile and not gpu")


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("backend,torch_group", GPU_BACKEND_VARIANTS)
def test_torch_gpu(session: nox.Session, backend: str, torch_group: str) -> None:
    """Run torch frontend tests on GPU backends (CUDA/ROCm)."""

    _skip_if_backend_unavailable(session, backend)

    _install(session, project=True, groups=["test", torch_group])

    session.run("pytest", "-q", "-m", "torch and gpu and not compile")


@nox.session(python="3.12")
def test_torch_cu118_min(session: nox.Session) -> None:
    """Run CUDA torch frontend tests with minimum deps (Python 3.12 only)."""

    _skip_if_backend_unavailable(session, "cuda")
    _install(
        session,
        project=True,
        groups=["torch-cu118"],
        resolution="lowest-direct",
    )
    _install(session, groups=["test"])
    _show_versions(session, torch=True)

    session.run("pytest", "-q", "-m", "torch and gpu and not compile")


@nox.session(python=PYTHON_VERSIONS)
def test_torch_compile_cpu(session: nox.Session) -> None:
    """Run CPU-only torch.compile tests."""

    _skip_if_torch_compile_cpu_unavailable(session)

    _install(session, project=True, groups=["test", "torch-cpu"])

    session.run("pytest", "-q", "-m", "compile and not gpu")


@nox.session(python="3.12")
def test_torch_compile_cpu_min(session: nox.Session) -> None:
    """Run CPU-only torch.compile tests with minimum deps (Python 3.12 only)."""

    _skip_if_torch_compile_cpu_unavailable(session)

    _install(
        session,
        project=True,
        groups=["torch-cpu"],
        resolution="lowest-direct",
    )
    _install(session, groups=["test"])
    _show_versions(session, torch=True)

    session.run("pytest", "-q", "-m", "compile and not gpu")


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("backend,torch_group", GPU_BACKEND_VARIANTS)
def test_torch_compile_gpu(
    session: nox.Session, backend: str, torch_group: str
) -> None:
    """Run torch.compile tests on GPU backends (CUDA/ROCm, opt-in)."""

    _skip_if_backend_unavailable(session, backend)

    _install(session, project=True, groups=["test", torch_group])

    session.run("pytest", "-q", "-m", "compile and gpu")


@nox.session(python="3.12")
def test_torch_compile_cu118_min(session: nox.Session) -> None:
    """Run CUDA torch.compile tests with minimum deps (Python 3.12 only, opt-in)."""

    _skip_if_backend_unavailable(session, "cuda")

    _install(
        session,
        project=True,
        groups=["torch-cu118"],
        resolution="lowest-direct",
    )
    _install(session, groups=["test"])
    _show_versions(session, torch=True)

    session.run("pytest", "-q", "-m", "compile and gpu")


@nox.session(python="3.12")
def docs(session: nox.Session) -> None:
    """Build documentation with Zensical."""
    _install(session, groups=["docs"])
    session.run("zensical", "build", "--strict", "--clean")
