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
        _install_project(session)
    except nox.command.CommandFailed:
        session.skip("Could not install runtime dependencies for this Python")
    _install(session, groups=["typecheck"])
    session.run("pyright", "src")
    # session.run("ty", "check", "src")
    # session.run("pyrefly", "check", "src")


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the unit tests against the default dependency resolver result."""
    try:
        _install_project(session)
    except nox.command.CommandFailed:
        session.skip("Could not install runtime dependencies for this Python")
    _install(session, groups=["test"])
    _show_versions(session)
    session.run("pytest", "-q")


@nox.session(python="3.12")
def tests_min(session: nox.Session) -> None:
    """Run tests with minimum supported numpy/numba (Python 3.12 only)."""
    # Project declares: numpy>=1.26, numba>=0.59
    _install(session, packages=["numpy==1.26.0", "numba==0.59.0"], groups=["test"])
    try:
        _install_project(session)
    except nox.command.CommandFailed:
        session.skip("Could not install project with minimum dependencies")
    _show_versions(session)
    session.run("pytest", "-q")


@nox.session(python="3.12")
def docs(session: nox.Session) -> None:
    """Build documentation with MkDocs."""
    # Material for MkDocs prints a warning about MkDocs 2.0; we intentionally
    # stay on MkDocs 1.x for now.
    session.env["NO_MKDOCS_2_WARNING"] = "1"
    _install(session, groups=["docs"])
    session.run("mkdocs", "build", "--strict")
