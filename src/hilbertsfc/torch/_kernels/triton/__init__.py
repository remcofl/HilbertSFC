"""Triton kernels for the torch frontend.

This subpackage imports `triton` at module import time and is therefore imported
lazily by the public API only when a Triton backend is selected.
"""
