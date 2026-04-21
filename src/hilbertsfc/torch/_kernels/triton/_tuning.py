"""Helpers for Triton launch configuration tuning."""

import triton  # type: ignore[reportMissingImports]

from ..._tuning_mode import TritonTuningMode, validate_triton_tuning_mode

# Keep a broad but bounded search space. AMD can prefer very different points
# for neighboring sizes, so include both low-warp and high-warp variants.
_AUTOTUNE_CONFIGS: tuple[tuple[int, int], ...] = (
    (128, 1),
    (128, 2),
    (128, 4),
    (256, 2),
    (256, 4),
    (512, 4),
    (512, 8),
    (1024, 4),
    (1024, 8),
    (2048, 4),
    (2048, 8),
    (4096, 4),
)

_TRITON_AUTOTUNE_CONFIGS: tuple[triton.Config, ...] = tuple(
    triton.Config({"BLOCK_SIZE": block_size}, num_warps=num_warps)
    for block_size, num_warps in _AUTOTUNE_CONFIGS
)


def _bucketed_size_key(n_elements: int) -> int:
    """Return a coarse key for autotune cache lookup.

    Policy:
    - Up to 64Ki elements: a single bucket.
    - Above 64Ki: split each power-of-two range into two buckets.
    """
    _64Ki = 1 << 16  # noqa: N806
    if n_elements <= _64Ki:
        return _64Ki

    lo = 1 << (n_elements.bit_length() - 1)
    mid = lo + (lo >> 1)
    return mid if n_elements < mid else lo << 1


def validate_tuning_mode(tuning: str) -> TritonTuningMode:
    return validate_triton_tuning_mode(tuning)


def triton_autotune_configs() -> tuple[triton.Config, ...]:
    """Return the shared Triton autotune config set."""

    return _TRITON_AUTOTUNE_CONFIGS


def autotune_key_for_elements(n_elements: int, *, tuning: TritonTuningMode) -> int:
    """Map element count to autotune key according to the selected mode."""

    if tuning == "autotune_exact":
        return n_elements
    return _bucketed_size_key(n_elements)
