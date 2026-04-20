from pathlib import Path

import pytest

from spectral_research.io_contract import (
    model_zoo_analysis_run_path,
    model_zoo_curated_table_path,
    model_zoo_root,
)


def test_model_zoo_root_points_to_nested_repo():
    expected = Path(__file__).resolve().parents[1] / "Model-Zoo"
    assert model_zoo_root() == expected


def test_model_zoo_curated_table_path_points_to_canonical_location():
    path = model_zoo_curated_table_path()
    assert path == model_zoo_root() / "data" / "curated" / "model_zoo_phase2.csv"
    assert path.is_file()


def test_model_zoo_analysis_run_path_builds_phase2_run_root():
    path = model_zoo_analysis_run_path("example_run")
    assert path == model_zoo_root() / "analysis_runs" / "phase2" / "example_run"


@pytest.mark.parametrize("run_name", ["", ".", "..", "nested/run", "nested\\run"])
def test_model_zoo_analysis_run_path_rejects_traversal_like_values(run_name):
    with pytest.raises(ValueError, match="single path component"):
        model_zoo_analysis_run_path(run_name)
