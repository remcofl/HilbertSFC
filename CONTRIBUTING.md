# Contributing

Thanks for contributing to **HilbertSFC**.

## Proposing changes (issue first)

To keep maintenance manageable (and to avoid drive-by automated/AI-generated PRs), this project generally requires an issue before a pull request.

- **Open an issue first** for anything beyond a tiny typo/doc fix.
- Wait for maintainers to triage the issue and apply the **`actionable`** label.
- Only then open a PR that references the issue (e.g. `Fixes #123` or `Refs #123`).

If you're unsure whether something warrants an issue first, open an issue anyway. This is the fastest way to confirm direction.

### About AI-assisted changes

Using AI tools while developing is fine, but:

- Please don't open PRs that are effectively "generated and dumped" without fully understanding the changes.
- You are responsible for the code you submit (correctness, tests, style).
- Maintainers may close PRs that don't follow the issue-first / `actionable` flow.

### What makes an issue `actionable`

Typically an `actionable` issue has:

- a clear problem statement,
- agreement on the expected behavior / API,
- and at least a rough approach (or acceptance criteria).

## Development workflow

This repo uses **uv** for environment management and **nox** for repeatable task automation.

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
- Core unit tests (runs the test session for the configured Python versions):
  - `uvx nox -s test`
- Minimum-dependency test run (Python 3.12 only):
  - `uvx nox -s test_min`
- Unit tests for torch submodule (CPU PyTorch only):
  - `uvx nox -s test_torch_cpu`
- Docs build (MkDocs strict build):
  - `uvx nox -s docs`

For a list of all sessions, run `uvx nox --list`.

## Type hints and public API stubs

Public function signatures are typed via stub files (`.pyi`) in the package directory.

- If you change a public signature in `src/hilbertsfc/hilbert2d.py` or `src/hilbertsfc/hilbert3d.py`, update the matching stub:
  - `src/hilbertsfc/hilbert2d.pyi`
  - `src/hilbertsfc/hilbert3d.pyi`
The same applies to `morton2d.py` and `morton3d.py` and their stubs.
- `src/hilbertsfc/py.typed` marks the package as typed (PEP 561); keep it included when packaging.

## Documentation

[Documentation](https://remcofl.github.io/HilbertSFC/) is hosted online. It includes a quick start guide, advanced usage, and API reference.

To serve the docs locally:

```bash
uv run --no-dev --group docs mkdocs serve
```

Build a static site into `site/`:

```bash
uv run --no-dev --group docs mkdocs build
```
