from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from spectral_research.study.contracts import RunSettings


_PROTOCOL_FIELDS = frozenset(
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

    actual_fields = set(payload)
    missing = sorted(_PROTOCOL_FIELDS - actual_fields)
    unexpected = sorted(actual_fields - _PROTOCOL_FIELDS)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected fields: {', '.join(unexpected)}")
        raise ValueError("invalid protocol: " + "; ".join(details))

    values: dict[str, Any] = dict(payload)
    values["schema_version"] = values.pop("protocol_version")
    return StudyProtocol(**values)


__all__ = ["StudyProtocol", "load_protocol"]
