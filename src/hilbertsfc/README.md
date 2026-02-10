# hilbertsfc (internal)

## Contributor note

- Public function signatures are typed via stub files (`.pyi`) in the same package directory.
  If you change a public signature in `hilbert2d.py` / `hilbert3d.py`, update the matching
  stub (`hilbert2d.pyi` / `hilbert3d.pyi`) as well.
- `py.typed` marks the package as typed (PEP 561); keep it included when packaging.
