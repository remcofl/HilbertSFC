"""Shared Triton tuning mode definitions for torch frontend and kernels."""

from typing import Literal

TRITON_TUNING_MODES = ("heuristic", "autotune_bucketed", "autotune_exact")
type TritonTuningMode = Literal["heuristic", "autotune_bucketed", "autotune_exact"]
"""Triton launch tuning options for `hilbertsfc.torch` functions."""


def validate_triton_tuning_mode(triton_tuning: str) -> TritonTuningMode:
    if triton_tuning not in TRITON_TUNING_MODES:
        raise ValueError(
            f"triton_tuning must be one of: {TRITON_TUNING_MODES}; got {triton_tuning!r}"
        )
    return triton_tuning
