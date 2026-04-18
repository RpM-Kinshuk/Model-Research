from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from spectral_research.pipeline import write_benchmark_artifacts


class PipelineTests(unittest.TestCase):
    def test_pipeline_writes_auditable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            write_benchmark_artifacts(
                output_dir=out,
                audit_seed=pd.DataFrame([{"model_id": "seed-a"}]),
                retrieval_metrics=pd.DataFrame(
                    [
                        {
                            "method": "full_module_trace",
                            "slice": "all",
                            "top1_accuracy": 1.0,
                            "mrr": 1.0,
                            "recall@1": 1.0,
                            "recall@3": 1.0,
                            "recall@5": 1.0,
                        }
                    ]
                ),
                rejection_metrics=pd.DataFrame(
                    [{"method": "full_module_trace", "auroc": 1.0, "auprc": 1.0, "fpr95": 0.0}]
                ),
                risk_coverage=pd.DataFrame(
                    [{"method": "full_module_trace", "coverage": 1.0, "risk": 0.0, "threshold": 0.5}]
                ),
                gallery_supported=pd.DataFrame([{"model_id": "a0"}]),
                queries_supported=pd.DataFrame([{"model_id": "a1"}]),
                queries_unsupported=pd.DataFrame([{"model_id": "u0"}]),
                queries_heterogeneous=pd.DataFrame([{"model_id": "h0"}]),
                audited_labels=pd.DataFrame([{"model_id": "a0"}]),
                split_payload={"seed": 0},
                neighborhood=pd.DataFrame(
                    [{"method": "full_module_trace", "family": "fam-a", "neighbor": "fam-b", "rank": 1}]
                ),
            )
            self.assertTrue((out / "tier_a_audit_seed.csv").exists())
            self.assertTrue((out / "tier_a_audited_labels.csv").exists())
            self.assertTrue((out / "tier_a_splits.json").exists())
            self.assertTrue((out / "retrieval_metrics.csv").exists())
            self.assertTrue((out / "rejection_metrics.csv").exists())
            self.assertTrue((out / "risk_coverage.csv").exists())
            self.assertTrue((out / "family_neighborhood_topk.csv").exists())
            self.assertTrue((out / "report.md").exists())

    def test_cli_exposes_v2_arguments(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1] / "scripts" / "spectral_similarity_research.py"
        )
        spec = importlib.util.spec_from_file_location("spectral_similarity_research_cli", script_path)

        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        args = module.parse_args(
            [
                "--metrics-dir",
                "metrics",
                "--lineage-csv",
                "lineage.csv",
                "--leaderboard-csv",
                "leaderboard.csv",
                "--freeze-csv",
                "freeze.csv",
                "--output-dir",
                "out",
                "--min-family-size",
                "6",
                "--seed",
                "9",
            ]
        )

        self.assertEqual(args.metrics_dir, Path("metrics"))
        self.assertEqual(args.lineage_csv, Path("lineage.csv"))
        self.assertEqual(args.leaderboard_csv, Path("leaderboard.csv"))
        self.assertEqual(args.freeze_csv, Path("freeze.csv"))
        self.assertEqual(args.output_dir, Path("out"))
        self.assertEqual(args.min_family_size, 6)
        self.assertEqual(args.seed, 9)


if __name__ == "__main__":
    unittest.main()
