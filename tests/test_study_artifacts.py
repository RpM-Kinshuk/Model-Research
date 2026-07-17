from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from spectral_research.study.artifacts import (
    ArtifactValidationError,
    validate_artifact_manifest,
    write_artifact_manifest,
)


def _write_run(root: Path) -> Path:
    run = root / "htsr_test"
    run.mkdir()
    pd.DataFrame({"model_id": ["org/a", "org/b"], "value": [1, 2]}).to_csv(
        run / "table.csv", index=False
    )
    (run / "provenance.json").write_text(
        json.dumps({"run_id": run.name}), encoding="utf-8"
    )
    (run / "report.md").write_text("# Study\n", encoding="utf-8")
    write_artifact_manifest(run, run_id=run.name)
    return run


def test_artifact_manifest_records_digest_size_and_csv_shape(tmp_path: Path) -> None:
    run = _write_run(tmp_path)

    manifest = validate_artifact_manifest(
        run,
        expected_run_id=run.name,
        required_files=("table.csv", "provenance.json", "report.md"),
        validate_all=True,
    )

    table = manifest["artifacts"]["table.csv"]
    assert len(table["sha256"]) == 64
    assert table["size_bytes"] == (run / "table.csv").stat().st_size
    assert table["rows"] == 2
    assert table["columns"] == ["model_id", "value"]


def test_artifact_manifest_rejects_same_shape_content_mutation(tmp_path: Path) -> None:
    run = _write_run(tmp_path)
    table = run / "table.csv"
    table.write_text(table.read_text(encoding="utf-8").replace("org/a,1", "org/a,9"), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="digest mismatch.*table.csv"):
        validate_artifact_manifest(run, required_files=("table.csv",))


def test_artifact_manifest_rejects_wrong_run_identity(tmp_path: Path) -> None:
    run = _write_run(tmp_path)

    with pytest.raises(ArtifactValidationError, match="run_id"):
        validate_artifact_manifest(
            run, expected_run_id="htsr_different", required_files=("table.csv",)
        )


def test_artifact_manifest_can_validate_only_dashboard_files(tmp_path: Path) -> None:
    run = _write_run(tmp_path)
    (run / "report.md").write_text("changed outside dashboard scope\n", encoding="utf-8")

    validate_artifact_manifest(run, required_files=("table.csv",))
    with pytest.raises(ArtifactValidationError, match="mismatch.*report.md"):
        validate_artifact_manifest(
            run, required_files=("table.csv",), validate_all=True
        )
