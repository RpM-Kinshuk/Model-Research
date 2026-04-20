from spectral_research.io_contract import (
    model_zoo_analysis_run_path,
    model_zoo_curated_table_path,
    model_zoo_root,
)


def test_model_zoo_root_points_to_nested_repo():
    assert model_zoo_root().as_posix().endswith("Model-Zoo")


def test_model_zoo_curated_table_path_points_to_canonical_location():
    assert model_zoo_curated_table_path().as_posix().endswith(
        "Model-Zoo/data/curated/model_zoo_phase2.csv"
    )


def test_model_zoo_analysis_run_path_builds_phase2_run_root():
    assert model_zoo_analysis_run_path("example_run").as_posix().endswith(
        "Model-Zoo/analysis_runs/phase2/example_run"
    )
