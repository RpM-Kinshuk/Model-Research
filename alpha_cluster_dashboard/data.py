from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

import h5py
import numpy as np
import pandas as pd

COMMON_ALPHA_MODULES = (
    "mlp.down_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "self_attn.k_proj",
    "self_attn.o_proj",
    "self_attn.q_proj",
    "self_attn.v_proj",
)

CANONICAL_MODULE_ORDER = (
    "mlp_down",
    "mlp_gate",
    "mlp_up",
    "mlp_gate_up",
    "attn_k",
    "attn_o",
    "attn_q",
    "attn_qkv",
    "attn_v",
    "cross_attn_k",
    "cross_attn_o",
    "cross_attn_q",
    "cross_attn_v",
)

MODULE_ROLE_MAP = {
    "mlp.down_proj": "mlp_down",
    "output.dense": "mlp_down",
    "feed_forward.w2": "mlp_down",
    "mlp.fc2": "mlp_down",
    "mlp.gate_proj": "mlp_gate",
    "feed_forward.w1": "mlp_gate",
    "mlp.up_proj": "mlp_up",
    "intermediate.dense": "mlp_up",
    "feed_forward.w3": "mlp_up",
    "mlp.fc1": "mlp_up",
    "mlp.gate_up_proj": "mlp_gate_up",
    "self_attn.k_proj": "attn_k",
    "attention.self.key": "attn_k",
    "self_attn.qkv_proj_k": "attn_k",
    "self_attn.k_proj_k": "attn_k",
    "self_attn.v_proj_k": "attn_k",
    "self_attn.o_proj": "attn_o",
    "self_attn.out_proj": "attn_o",
    "attention.output.dense": "attn_o",
    "attention.wo": "attn_o",
    "self_attn.q_proj": "attn_q",
    "attention.self.query": "attn_q",
    "self_attn.qkv_proj_q": "attn_q",
    "self_attn.k_proj_q": "attn_q",
    "self_attn.v_proj_q": "attn_q",
    "attention.wqkv": "attn_qkv",
    "self_attn.k_proj_v": "attn_v",
    "self_attn.qkv_proj_v": "attn_v",
    "self_attn.v_proj": "attn_v",
    "self_attn.v_proj_v": "attn_v",
    "attention.self.value": "attn_v",
    "cross_attn.k_proj": "cross_attn_k",
    "cross_attn.o_proj": "cross_attn_o",
    "cross_attn.q_proj": "cross_attn_q",
    "cross_attn.v_proj": "cross_attn_v",
}

CANONICAL_MODULE_LABELS = {
    "mlp_down": "MLP down",
    "mlp_gate": "MLP gate",
    "mlp_up": "MLP up",
    "mlp_gate_up": "MLP gate+up",
    "attn_k": "Attention K",
    "attn_o": "Attention O",
    "attn_q": "Attention Q",
    "attn_qkv": "Attention QKV",
    "attn_v": "Attention V",
    "cross_attn_k": "Cross-attn K",
    "cross_attn_o": "Cross-attn O",
    "cross_attn_q": "Cross-attn Q",
    "cross_attn_v": "Cross-attn V",
}

CSV_PROJECTION_MAP = {
    "down": "mlp.down_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "key": "self_attn.k_proj",
    "output": "self_attn.o_proj",
    "query": "self_attn.q_proj",
    "value": "self_attn.v_proj",
    # These rows are usually single global heads rather than per-layer traces.
    "lm_head": None,
}

CSV_LONGNAME_PATTERNS = (
    ("self_attn.k_proj", "self_attn.k_proj"),
    ("self_attn.o_proj", "self_attn.o_proj"),
    ("self_attn.q_proj", "self_attn.q_proj"),
    ("self_attn.v_proj", "self_attn.v_proj"),
    ("cross_attn.k_proj", "cross_attn.k_proj"),
    ("cross_attn.o_proj", "cross_attn.o_proj"),
    ("cross_attn.q_proj", "cross_attn.q_proj"),
    ("cross_attn.v_proj", "cross_attn.v_proj"),
    ("mlp.down_proj", "mlp.down_proj"),
    ("mlp.gate_proj", "mlp.gate_proj"),
    ("mlp.up_proj", "mlp.up_proj"),
    ("mlp.gate_up_proj", "mlp.gate_up_proj"),
    ("feed_forward.w1", "feed_forward.w1"),
    ("feed_forward.w2", "feed_forward.w2"),
    ("feed_forward.w3", "feed_forward.w3"),
    ("attention.self.key", "attention.self.key"),
    ("attention.self.query", "attention.self.query"),
    ("attention.self.value", "attention.self.value"),
    ("attention.output.dense", "attention.output.dense"),
    ("attention.wo", "attention.wo"),
    ("attention.wqkv", "attention.wqkv"),
    ("intermediate.dense", "intermediate.dense"),
    ("output.dense", "output.dense"),
)

CSV_LAYER_PREFIX_PATTERNS = (
    r"^model\.layers\.\d+\.",
    r"^layers\.\d+\.",
    r"^decoder\.layers\.\d+\.",
    r"^model\.decoder\.layers\.\d+\.",
    r"^encoder\.layers\.\d+\.",
    r"^model\.encoder\.layers\.\d+\.",
    r"^transformer\.h\.\d+\.",
    r"^gpt_neox\.layers\.\d+\.",
    r"^bert\.encoder\.layer\.\d+\.",
    r"^encoder\.layer\.\d+\.",
    r"^h\.\d+\.",
)


@dataclass(frozen=True)
class AlphaRecord:
    model_id: str
    display_name: str
    path: str
    num_layers: int
    raw_modules: tuple[str, ...]
    alpha: np.ndarray

    @property
    def common_schema(self) -> bool:
        return self.raw_modules == COMMON_ALPHA_MODULES

    @property
    def raw_module_count(self) -> int:
        return len(self.raw_modules)


def canonicalize_module_name(module_name: str) -> str:
    return MODULE_ROLE_MAP.get(module_name, f"other::{module_name}")


def list_metric_files(metrics_dir: str | Path) -> list[Path]:
    root = Path(metrics_dir).expanduser()
    return sorted(list(root.glob("*.h5")) + list(root.glob("*.csv")))


def _csv_usecols(column_name: str) -> bool:
    return column_name in {"model_id", "projection", "longname", "layer", "alpha"}


def _strip_csv_layer_prefix(module_name: str) -> str:
    normalized = module_name.strip()
    for pattern in CSV_LAYER_PREFIX_PATTERNS:
        normalized = re.sub(pattern, "", normalized)
    return normalized


def _normalize_csv_module_name(projection: object, longname: object) -> str | None:
    for candidate in (projection, longname):
        if not isinstance(candidate, str):
            continue
        stripped = candidate.strip()
        if not stripped:
            continue
        if stripped in CSV_PROJECTION_MAP:
            return CSV_PROJECTION_MAP[stripped]

        normalized = _strip_csv_layer_prefix(stripped)
        if normalized in CSV_PROJECTION_MAP:
            return CSV_PROJECTION_MAP[normalized]
        if normalized in MODULE_ROLE_MAP:
            return normalized
        for pattern, module_name in CSV_LONGNAME_PATTERNS:
            if pattern in normalized:
                return module_name
    return None


def _ordered_raw_modules(module_names: Iterable[str]) -> tuple[str, ...]:
    unique = [name for name in dict.fromkeys(module_names) if name]
    if set(unique) == set(COMMON_ALPHA_MODULES):
        return COMMON_ALPHA_MODULES

    canonical_order = {name: index for index, name in enumerate(CANONICAL_MODULE_ORDER)}

    def sort_key(module_name: str) -> tuple[int, int, str]:
        canonical_name = canonicalize_module_name(module_name)
        primary = 0 if canonical_name in canonical_order else 1
        secondary = canonical_order.get(canonical_name, len(canonical_order))
        return primary, secondary, module_name

    return tuple(sorted(unique, key=sort_key))


def _load_h5_alpha_record(path: Path) -> AlphaRecord | None:
    with h5py.File(path, "r") as handle:
        if "alpha" not in handle:
            return None
        dataset = handle["alpha"]
        modules = tuple(json.loads(dataset.attrs["module_names_json"]))
        num_layers = int(dataset.attrs["num_layers"])
        alpha = np.asarray(dataset[:], dtype=float)
        alpha[~np.isfinite(alpha)] = np.nan
    stem = path.stem
    return AlphaRecord(
        model_id=stem,
        display_name=stem.replace("--", "/"),
        path=str(path),
        num_layers=num_layers,
        raw_modules=modules,
        alpha=alpha,
    )


def _load_csv_alpha_record(path: Path) -> AlphaRecord | None:
    try:
        frame = pd.read_csv(path, usecols=_csv_usecols)
    except (pd.errors.EmptyDataError, FileNotFoundError, ValueError):
        return None

    required_columns = {"projection", "layer", "alpha"}
    if frame.empty or not required_columns.issubset(frame.columns):
        return None

    if "longname" not in frame.columns:
        frame["longname"] = np.nan
    frame["raw_module"] = [
        _normalize_csv_module_name(projection, longname)
        for projection, longname in zip(frame["projection"], frame["longname"], strict=False)
    ]
    frame["layer"] = pd.to_numeric(frame["layer"], errors="coerce")
    frame["alpha"] = pd.to_numeric(frame["alpha"], errors="coerce")
    frame = frame.dropna(subset=["raw_module", "layer", "alpha"]).copy()
    if frame.empty:
        return None

    frame["layer"] = frame["layer"].astype(int)
    grouped = (
        frame.groupby(["layer", "raw_module"], as_index=False)["alpha"]
        .mean()
        .sort_values(["layer", "raw_module"])
        .reset_index(drop=True)
    )
    if grouped.empty:
        return None

    modules = _ordered_raw_modules(grouped["raw_module"].tolist())
    module_to_index = {module_name: index for index, module_name in enumerate(modules)}
    max_layer = int(grouped["layer"].max())
    alpha = np.full((max_layer + 1, len(modules)), np.nan, dtype=float)
    for row in grouped.itertuples(index=False):
        alpha[int(row.layer), module_to_index[row.raw_module]] = float(row.alpha)
    alpha[~np.isfinite(alpha)] = np.nan

    display_name = path.stem.replace("--", "/")
    if "model_id" in frame.columns:
        model_ids = frame["model_id"].dropna().astype(str)
        if not model_ids.empty:
            display_name = model_ids.iloc[0]

    return AlphaRecord(
        model_id=path.stem,
        display_name=display_name,
        path=str(path),
        num_layers=max_layer + 1,
        raw_modules=modules,
        alpha=alpha,
    )


def load_alpha_records(metrics_dir: str | Path) -> list[AlphaRecord]:
    records: list[AlphaRecord] = []
    for path in list_metric_files(metrics_dir):
        if path.suffix == ".h5":
            record = _load_h5_alpha_record(path)
        elif path.suffix == ".csv":
            record = _load_csv_alpha_record(path)
        else:
            record = None
        if record is not None:
            records.append(record)
    return records


def schema_name(record: AlphaRecord) -> str:
    if record.common_schema:
        return "common_llm_7"
    canonical_modules = sorted({canonicalize_module_name(module) for module in record.raw_modules})
    return " + ".join(canonical_modules)


def records_to_frame(records: Iterable[AlphaRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "model_id": record.model_id,
                "display_name": record.display_name,
                "path": record.path,
                "num_layers": record.num_layers,
                "raw_module_count": record.raw_module_count,
                "raw_modules": ", ".join(record.raw_modules),
                "common_schema": record.common_schema,
                "schema_name": schema_name(record),
            }
        )
    return pd.DataFrame(rows)


def available_modules(records: Iterable[AlphaRecord], schema_mode: str) -> list[str]:
    seen: dict[str, int] = {}
    for record in records:
        if schema_mode == "common":
            if not record.common_schema:
                continue
            modules = record.raw_modules
        else:
            modules = tuple(canonicalize_module_name(module) for module in record.raw_modules)
        for module in modules:
            seen[module] = seen.get(module, 0) + 1

    if schema_mode == "common":
        ordered = [module for module in COMMON_ALPHA_MODULES if module in seen]
    else:
        ordered = [module for module in CANONICAL_MODULE_ORDER if module in seen]
        ordered.extend(sorted(module for module in seen if module not in ordered))
    return ordered


def module_label(module_name: str) -> str:
    return CANONICAL_MODULE_LABELS.get(module_name, module_name.replace("_", " ").replace("::", ": "))
