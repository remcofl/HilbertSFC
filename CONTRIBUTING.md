# Contributing

Thanks for contributing to **HilbertSFC**.

## Development workflow

This repo uses **uv** for environment management and **nox** for repeatable task automation (`noxfile.py` sets `default_venv_backend = "uv"`).

For quick one-off checks (or to match CI closely), `uvx nox` is convenient.

For any more serious work (iterating on code, working on notebooks/docs), you should sync a local environment (`uv sync`).

### Recommended: sync a local environment

- Create/sync the local environment (dev dependencies are included by default):
  - `uv sync`

- Run individual tools directly in that environment:
  - `uv run pytest -q`
  - `uv run pyright src`

- Run formatting and linting:
  - `uvx ruff check src`
  - `uvx ruff format --check src`

If you're working on notebooks, docs, benches, or scripts, you can include optional dependency groups:

- Notebook dependencies (Jupyter, matplotlib, etc.):
  - `uv sync --group notebooks`
- Documentation dependencies (MkDocs, mkdocstrings, etc.):
  - `uv sync --group docs`

### CI tasks with nox

- Lint + format check:
  - `uvx nox -s lint`
- Type check:
  - `uvx nox -s typecheck`
- Unit tests (runs the test session for the configured Python versions):
  - `uvx nox -s tests`
- Minimum-dependency test run (Python 3.12 only):
  - `uvx nox -s tests_min`
- Docs build (MkDocs strict build):
  - `uvx nox -s docs`

## Type hints and public API stubs

Public function signatures are typed via stub files (`.pyi`) in the package directory.

- If you change a public signature in `src/hilbertsfc/hilbert2d.py` or `src/hilbertsfc/hilbert3d.py`, update the matching stub:
  - `src/hilbertsfc/hilbert2d.pyi`
  - `src/hilbertsfc/hilbert3d.pyi`
- `src/hilbertsfc/py.typed` marks the package as typed (PEP 561); keep it included when packaging.
