from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from spectral_research.study.contracts import RunSettings


_BASE_PROTOCOL_FIELDS = frozenset(
    {
        "protocol_version",
        "random_seed",
        "bootstrap_replicates",
        "signflip_replicates",
        "outer_folds",
        "inner_folds",
        "ridge_alphas",
        "minimum_components",
        "minimum_lineage_families",
        "min_feature_coverage",
        "min_pair_coverage",
        "min_role_coverage",
        "cluster_count_rule",
        "external_evidence_role",
    }
)
_HOLDOUT_PROTOCOL_FIELDS = frozenset(
    {"holdout_repeats", "holdout_test_fraction"}
)


@dataclass(frozen=True)
class StudyProtocol(RunSettings):
    """Versioned confirmatory settings accepted by the study runner."""

    @property
    def protocol_version(self) -> str:
        return self.schema_version


def load_protocol(path: str | Path) -> StudyProtocol:
    """Load and validate one complete JSON study protocol."""

    protocol_path = Path(path)
    with protocol_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("protocol JSON must contain an object")

    version = payload.get("protocol_version")
    if version == "htsr-combined-study-v3":
        expected_fields = _BASE_PROTOCOL_FIELDS
    elif version == "htsr-combined-study-v4":
        expected_fields = _BASE_PROTOCOL_FIELDS | _HOLDOUT_PROTOCOL_FIELDS
    else:
        raise ValueError(f"unsupported protocol_version: {version!r}")

    actual_fields = set(payload)
    missing = sorted(expected_fields - actual_fields)
    unexpected = sorted(actual_fields - expected_fields)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected fields: {', '.join(unexpected)}")
        raise ValueError("invalid protocol: " + "; ".join(details))

    values: dict[str, Any] = dict(payload)
    values["schema_version"] = values.pop("protocol_version")
    if version == "htsr-combined-study-v3":
        values["holdout_repeats"] = 0
        values["holdout_test_fraction"] = 0.3
    return StudyProtocol(**values)


__all__ = ["StudyProtocol", "load_protocol"]
