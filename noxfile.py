import nox
import nox.command

# Keep CI output readable.
nox.options.reuse_existing_virtualenvs = False
nox.options.stop_on_first_error = False
nox.options.default_venv_backend = "uv"

PYTHON_VERSIONS: tuple[str, ...] = ("3.12", "3.13", "3.14")


def _install(
    session: nox.Session,
    *args: str,
    packages: list[str] | None = None,
    groups: list[str] | None = None,
) -> None:
    install_args = list(args)
    if packages is not None:
        install_args.extend(packages)
    if groups is not None:
        for group in groups:
            install_args.extend(["--group", group])
    session.install(*install_args)


def _install_project(session: nox.Session) -> None:
    """Install this project (editable)."""
    session.install("-e", ".")


def _show_versions(session: nox.Session) -> None:
    session.run(
        "python",
        "-c",
        "import numpy, numba; print('numpy', numpy.__version__); print('numba', numba.__version__)",
    )


@nox.session(venv_backend="none")
def lint(session: nox.Session) -> None:
    """Run Ruff (lint + format check)."""
    session.run("uvx", "ruff", "check", "src", external=True)
    session.run("uvx", "ruff", "format", "--check", "src", external=True)


@nox.session(python=PYTHON_VERSIONS)
def typecheck(session: nox.Session) -> None:
    """Run type check."""
    try:
        _install(session, groups=["typecheck", "torch-cpu"])
        _install_project(session)
    except nox.command.CommandFailed:
        session.skip("Could not install runtime dependencies for this Python")
    session.run("pyright", "src")
    # Faster alternatives, but still gaps in type resolution:
    # session.run("pyrefly", "check", "src")
    # session.run("ty", "check", "src")


@nox.session(python=PYTHON_VERSIONS)
def test(session: nox.Session) -> None:
    """Run the core unit tests against the default dependency resolver result."""
    try:
        _install(session, groups=["test"])
        _install_project(session)
    except nox.command.CommandFailed:
        session.skip("Could not install runtime dependencies for this Python")
    _show_versions(session)
    session.run("pytest", "-q", "-n", "auto", "-m", "not torch")


@nox.session(python="3.12")
def test_min(session: nox.Session) -> None:
    """Run core unit tests with minimum supported numpy/numba (Python 3.12 only)."""
    # Project declares: numpy>=1.26, numba>=0.59
    try:
        _install(session, groups=["test", "runtime-min"])
        _install_project(session)
    except nox.command.CommandFailed:
        session.skip("Could not install project with minimum dependencies")
    _show_versions(session)
    session.run("pytest", "-q", "-n", "auto", "-m", "not torch")


@nox.session(python=PYTHON_VERSIONS)
def test_torch_cpu(session: nox.Session) -> None:
    """Run torch frontend tests that are CPU-only and fast enough for regular CI."""

    try:
        _install(session, groups=["test", "torch-cpu"])
        _install_project(session)
    except nox.command.CommandFailed:
        session.skip("Could not install project with torch-cpu-* dependencies")

    session.run("pytest", "-q", "-m", "torch and not gpu")


@nox.session(python="3.12")
def test_torch_cpu_min(session: nox.Session) -> None:
    """Run torch frontend tests that are CPU-only and fast enough for regular CI."""

    try:
        _install(session, groups=["test", "runtime-min", "torch-cpu-min"])
        _install_project(session)
    except nox.command.CommandFailed:
        session.skip("Could not install project with torch-cpu-* dependencies")

    session.run("pytest", "-q", "-m", "torch and not gpu")


@nox.session(python=PYTHON_VERSIONS)
def test_torch_cu130(session: nox.Session) -> None:
    """Run CUDA/Triton torch tests (intended for dedicated GPU runners)."""

    try:
        _install(session, groups=["test", "torch-cu130"])
        _install_project(session)
    except nox.command.CommandFailed:
        session.skip("Could not install project with torch-cu130 dependencies")

    session.run("pytest", "-q", "-m", "torch and gpu")


@nox.session(python="3.12")
def test_torch_cu118_min(session: nox.Session) -> None:
    """Run CUDA/Triton torch tests (intended for dedicated GPU runners)."""

    try:
        _install(session, groups=["test", "torch-cu118-min"])
        _install_project(session)
    except nox.command.CommandFailed:
        session.skip("Could not install project with torch-cu118-min dependencies")

    session.run("pytest", "-q", "-m", "torch and gpu")


@nox.session(python=PYTHON_VERSIONS)
def test_torch_compile(session: nox.Session) -> None:
    """Run torch.compile tests (opt-in; slower and potentially brittle)."""

    try:
        _install(session, groups=["test", "torch-cpu"])
        _install_project(session)
    except nox.command.CommandFailed:
        session.skip("Could not install project with torch-cpu-* dependencies")

    session.run("pytest", "-q", "-m", "compile and not gpu")


@nox.session(python="3.12")
def test_torch_compile_min(session: nox.Session) -> None:
    """Run torch.compile tests (opt-in; slower and potentially brittle)."""

    try:
        _install(session, groups=["test", "runtime-min", "torch-cpu-min"])
        _install_project(session)
    except nox.command.CommandFailed:
        session.skip("Could not install project with torch-cpu-* dependencies")

    session.run("pytest", "-q", "-m", "compile and not gpu")


@nox.session(python=PYTHON_VERSIONS)
def test_torch_compile_cu130(session: nox.Session) -> None:
    """Run torch.compile tests (opt-in; slower and potentially brittle)."""

    try:
        _install(session, groups=["test", "torch-cu130"])
        _install_project(session)
    except nox.command.CommandFailed:
        session.skip("Could not install project with torch-cu130 dependencies")

    session.run("pytest", "-q", "-m", "compile and gpu")


@nox.session(python="3.12")
def test_torch_compile_cu118_min(session: nox.Session) -> None:
    """Run torch.compile tests (opt-in; slower and potentially brittle)."""

    try:
        _install(session, groups=["test", "runtime-min", "torch-cu118-min"])
        _install_project(session)
    except nox.command.CommandFailed:
        session.skip("Could not install project with torch-cu118-min dependencies")

    session.run("pytest", "-q", "-m", "compile and gpu")


@nox.session(python="3.12")
def docs(session: nox.Session) -> None:
    """Build documentation with MkDocs."""
    # Material for MkDocs prints a warning about MkDocs 2.0; we intentionally
    # stay on MkDocs 1.x for now.
    session.env["NO_MKDOCS_2_WARNING"] = "1"
    _install(session, groups=["docs"])
    session.run("mkdocs", "build", "--strict")
