from __future__ import annotations

import os
from pathlib import Path

from alpha_cluster_dashboard import paths


def _complete_run(root: Path, name: str, markers: tuple[str, ...], timestamp: int) -> Path:
    run = root / name
    run.mkdir(parents=True)
    for marker in markers:
        artifact = run / marker
        artifact.write_text("{}\n", encoding="utf-8")
        os.utime(artifact, (timestamp, timestamp))
    return run


def test_default_run_paths_choose_latest_complete_local_runs(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("HTSR_STUDY_RUN", raising=False)
    monkeypatch.delenv("HTSR_SCALEUP_RUN", raising=False)

    study_root = tmp_path / "research_runs" / "htsr_study" / "runs"
    old_study = _complete_run(study_root, "htsr_old", paths.STUDY_RUN_MARKERS, 10)
    new_study = _complete_run(study_root, "htsr_new", paths.STUDY_RUN_MARKERS, 20)
    incomplete = study_root / "htsr_incomplete"
    incomplete.mkdir()
    os.utime(incomplete, (30, 30))

    scale_root = tmp_path / "research_runs" / "htsr_weight_only" / "runs"
    old_scale = _complete_run(scale_root, "scaleup_old", paths.SCALEUP_RUN_MARKERS, 10)
    new_scale = _complete_run(scale_root, "scaleup_new", paths.SCALEUP_RUN_MARKERS, 20)

    assert old_study != new_study
    assert old_scale != new_scale
    assert paths.default_study_run() == new_study
    assert paths.default_scaleup_run() == new_scale


def test_explicit_run_environment_overrides_local_discovery(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    study = tmp_path / "explicit-study"
    scaleup = tmp_path / "explicit-scaleup"
    monkeypatch.setenv("HTSR_STUDY_RUN", str(study))
    monkeypatch.setenv("HTSR_SCALEUP_RUN", str(scaleup))

    assert paths.default_study_run() == study
    assert paths.default_scaleup_run() == scaleup
