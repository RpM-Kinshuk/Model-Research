import json
from pathlib import Path

import pandas as pd

from spectral_research.ablation_analysis import (
    infer_coarse_family,
    load_terminal_statuses,
    safe_stem_to_model_id,
    summarize_model_stats,
    summarize_run_coverage,
)


def test_safe_stem_to_model_id_restores_worker_filename_separator():
    assert safe_stem_to_model_id("Owner--Model-7B") == "Owner/Model-7B"
    assert safe_stem_to_model_id("gpt2") == "gpt2"


def test_load_terminal_statuses_reads_json_records(tmp_path: Path):
    status_dir = tmp_path / "logs" / "terminal_status"
    status_dir.mkdir(parents=True)
    (status_dir / "org--ok.json").write_text(
        json.dumps(
            {
                "ts": "2026-07-01T00:00:00+00:00",
                "model_id": "org/ok",
                "status": "success",
                "stage": "save",
                "reason": "completed",
                "message": "done",
                "attempt": 1,
            }
        )
        + "\n"
    )

    frame = load_terminal_statuses(tmp_path)

    assert frame.loc[0, "model_id"] == "org/ok"
    assert frame.loc[0, "status"] == "success"
    assert frame.loc[0, "reason"] == "completed"


def test_summarize_model_stats_groups_alpha_by_canonical_role(tmp_path: Path):
    stats_path = tmp_path / "org--model.csv"
    pd.DataFrame(
        [
            {
                "model_id": "org/model",
                "longname": "model.layers.0.self_attn.q_proj",
                "alpha": 1.5,
                "log_norm": 2.0,
                "spectral_norm": 10.0,
            },
            {
                "model_id": "org/model",
                "longname": "model.layers.1.self_attn.q_proj",
                "alpha": 2.5,
                "log_norm": 4.0,
                "spectral_norm": 20.0,
            },
            {
                "model_id": "org/model",
                "longname": "model.layers.0.mlp.down_proj",
                "alpha": 4.0,
                "log_norm": 8.0,
                "spectral_norm": 40.0,
            },
        ]
    ).to_csv(stats_path, index=False)

    model_row, role_frame = summarize_model_stats(stats_path)

    assert model_row["model_id"] == "org/model"
    assert model_row["alpha_global_mean"] == 8.0 / 3.0
    assert model_row["num_layers_observed"] == 2
    assert model_row["alpha_attn_mean"] == 2.0
    assert model_row["alpha_mlp_mean"] == 4.0
    assert model_row["alpha_mlp_minus_attn"] == 2.0
    assert set(role_frame["module_role"]) == {"attn_q", "mlp_down"}


def test_summarize_run_coverage_joins_selected_status_and_artifacts():
    selected = pd.DataFrame(
        [
            {
                "model_id": "org/ok",
                "primary_type_bucket": "finetune",
                "lineage_eval_group": "same_base_finetune",
                "source_model": "meta-llama/Llama-3-8B",
                "Architecture": "LlamaForCausalLM",
            },
            {
                "model_id": "org/fail",
                "primary_type_bucket": "moe",
                "lineage_eval_group": "unrelated_negative_pair",
                "source_model": "unknown",
                "Architecture": "MixtralForCausalLM",
            },
        ]
    )
    statuses = pd.DataFrame(
        [
            {"model_id": "org/ok", "status": "success", "stage": "save", "reason": "completed"},
            {"model_id": "org/fail", "status": "failed", "stage": "load", "reason": "model_load_error"},
        ]
    )

    per_model, summary = summarize_run_coverage(
        selected,
        statuses,
        stats_model_ids={"org/ok"},
        metrics_model_ids={"org/ok"},
    )

    assert summary["selected_rows"] == 2
    assert summary["success_rows"] == 1
    assert summary["stats_artifact_rows"] == 1
    assert per_model.loc[per_model["model_id"] == "org/fail", "terminal_reason"].item() == "model_load_error"
    assert per_model.loc[per_model["model_id"] == "org/ok", "coarse_family"].item() == "llama"
    assert per_model.loc[per_model["model_id"] == "org/fail", "coarse_family"].item() == "mixtral"


def test_infer_coarse_family_prefers_known_source_then_architecture_hints():
    assert (
        infer_coarse_family(
            pd.Series(
                {
                    "model_id": "custom/chat",
                    "source_model": "Qwen/Qwen2-7B",
                    "Architecture": "Unknown",
                }
            )
        )
        == "qwen"
    )
    assert (
        infer_coarse_family(
            pd.Series(
                {
                    "model_id": "custom/chat",
                    "source_model": "unknown",
                    "Architecture": "GemmaForCausalLM",
                }
            )
        )
        == "gemma"
    )
