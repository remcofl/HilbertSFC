"""Torch ATen 'kernels'.

These kernels run with regular torch tensor operations and are mainly intended as a
portable fallback when Triton is unavailable.

Eager execution is slow (especially for large tensors), but with `torch.compile`
these can achieve near-Triton performance.
"""
