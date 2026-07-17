from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import htsr_combined_study
from spectral_research.study.contracts import RunSettings
from spectral_research.study.protocol import StudyProtocol, load_protocol


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PROTOCOL = REPO_ROOT / "configs" / "htsr_combined_study_v3.json"


def _required_cli_args(tmp_path: Path) -> list[str]:
    return [
        "--protocol",
        str(CANONICAL_PROTOCOL),
        "--metadata",
        str(tmp_path / "metadata.csv"),
        "--labels",
        str(tmp_path / "labels.csv"),
        "--graph",
        str(tmp_path / "graph.json"),
        "--development-stats",
        str(tmp_path / "development-stats"),
        "--development-targets",
        str(tmp_path / "development-targets.csv"),
        "--cache-root",
        str(tmp_path / "cache"),
        "--output-root",
        str(tmp_path / "runs"),
    ]


def test_canonical_confirmatory_protocol_is_explicit_and_versioned() -> None:
    protocol = load_protocol(CANONICAL_PROTOCOL)

    assert isinstance(protocol, StudyProtocol)
    assert isinstance(protocol, RunSettings)
    assert protocol.protocol_version == "htsr-combined-study-v3"
    assert protocol.to_dict() == {
        "schema_version": "htsr-combined-study-v3",
        "random_seed": 17,
        "bootstrap_replicates": 2_000,
        "signflip_replicates": 10_000,
        "outer_folds": 5,
        "inner_folds": 4,
        "ridge_alphas": [0.01, 0.1, 1.0, 10.0, 100.0],
        "minimum_components": 10,
        "minimum_lineage_families": 10,
        "min_feature_coverage": 0.8,
        "min_pair_coverage": 0.8,
        "min_role_coverage": 0.8,
        "cluster_count_rule": "sqrt_half",
        "external_evidence_role": "reused_descriptive",
    }


def test_protocol_loader_rejects_unrecognized_fields(tmp_path: Path) -> None:
    payload = json.loads(CANONICAL_PROTOCOL.read_text(encoding="utf-8"))
    payload["outcome_selected_knob"] = 1
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected fields.*outcome_selected_knob"):
        load_protocol(path)


def test_cli_loads_protocol_without_execution_setting_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run_study(inputs, settings, *, dry_run):
        captured.update(inputs=inputs, settings=settings, dry_run=dry_run)
        return SimpleNamespace(
            run_id="htsr_test",
            run_dir=None,
            dry_run=dry_run,
            cache_reused=False,
            summary={},
        )

    monkeypatch.setattr(htsr_combined_study, "run_study", fake_run_study)

    assert htsr_combined_study.main([*_required_cli_args(tmp_path), "--dry-run"]) == 0
    settings = captured["settings"]
    assert isinstance(settings, StudyProtocol)
    assert settings.random_seed == 17
    assert settings.bootstrap_replicates == 2_000
    assert captured["dry_run"] is True


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("--random-seed", "18"),
        ("--ridge-alphas", "0.1"),
        ("--min-role-coverage", "0.7"),
        ("--cluster-count-rule", "family_count"),
    ],
)
def test_cli_rejects_conflicting_execution_setting_overrides(
    tmp_path: Path,
    override: str,
    value: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        htsr_combined_study.main([*_required_cli_args(tmp_path), override, value])

    assert f"{override} conflicts with protocol" in capsys.readouterr().err


def test_cli_allows_repeated_matching_setting_for_legacy_invocations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        htsr_combined_study,
        "run_study",
        lambda inputs, settings, *, dry_run: SimpleNamespace(
            run_id="htsr_test",
            run_dir=None,
            dry_run=dry_run,
            cache_reused=False,
            summary={},
        ),
    )

    argv = [*_required_cli_args(tmp_path), "--random-seed", "17", "--dry-run"]
    assert htsr_combined_study.main(argv) == 0
