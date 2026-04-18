from __future__ import annotations

import numpy as np

ATTENTION_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")


def full_module_trace_features(alpha: np.ndarray) -> np.ndarray:
    return np.asarray(alpha, dtype=float)


def attention_trace_features(
    alpha: np.ndarray, modules: list[str] | tuple[str, ...]
) -> np.ndarray:
    keep = [index for index, module in enumerate(modules) if module.endswith(ATTENTION_SUFFIXES)]
    return np.asarray(alpha, dtype=float)[:, keep]


def alpha_scalar_features(alpha: np.ndarray) -> np.ndarray:
    arr = np.asarray(alpha, dtype=float)
    flat = arr.reshape(-1)
    return np.array(
        [[
            float(np.nanmean(flat)),
            float(np.nanstd(flat)),
            float(np.nanmedian(flat)),
            float(np.nanmean(arr[0])),
            float(np.nanmean(arr[-1])),
            float(np.nanmean(arr[-1]) - np.nanmean(arr[0])),
        ]]
    )
