"""Internal kernel implementations.

These modules hold the actual Numba kernels and their cached builder
functions. Public modules (hilbert2d/hilbert3d/morton2d/morton3d) call into builders via
hilbertsfc._dispatch.
"""
