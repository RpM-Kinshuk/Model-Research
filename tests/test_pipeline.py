from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from alpha_cluster_dashboard.data import AlphaRecord
from spectral_research.pipeline import FamilyPrototype, _score_query, write_benchmark_artifacts


class PipelineTests(unittest.TestCase):
    def _query_record(self) -> AlphaRecord:
        return AlphaRecord(
            model_id="query/a",
            display_name="query/a",
            path="/tmp/query-a",
            num_layers=2,
            raw_modules=("self_attn.q_proj", "mlp.down_proj"),
            alpha=np.array([[1.0, 2.0], [2.0, 3.0]], dtype=float),
        )

    def _prototype(self) -> FamilyPrototype:
        return FamilyPrototype(
            family="fam/a",
            modules=("self_attn.q_proj", "self_attn.k_proj", "mlp.down_proj"),
            full_alpha=np.array([[1.0, 9.0, 2.0], [2.0, 8.0, 3.0]], dtype=float),
            metadata_vector=np.array([9.0, 9.0], dtype=float),
            scalar_vector=np.array([0.0], dtype=float),
            summary_vector=np.array([0.0], dtype=float),
            architecture="LlamaForCausalLM",
            size_bucket="medium",
        )

    def test_score_query_supports_intrinsic_basics_only(self) -> None:
        row = pd.Series({"model_id": "query/a"})

        score = _score_query(
            method="intrinsic_basics_only",
            row=row,
            record=None,
            prototype=self._prototype(),
            metadata_vectors={"query/a": np.array([1.0, 2.0], dtype=float)},
            parent_scores=None,
        )

        self.assertTrue(np.isfinite(score))

    def test_score_query_supports_shared_role_trace(self) -> None:
        row = pd.Series({"model_id": "query/a"})

        score = _score_query(
            method="shared_role_trace",
            row=row,
            record=self._query_record(),
            prototype=self._prototype(),
            metadata_vectors={},
            parent_scores=None,
        )

        self.assertTrue(np.isfinite(score))

    def test_score_query_supports_role_aggregated_trace(self) -> None:
        row = pd.Series({"model_id": "query/a"})

        score = _score_query(
            method="role_aggregated_trace",
            row=row,
            record=self._query_record(),
            prototype=self._prototype(),
            metadata_vectors={},
            parent_scores=None,
        )

        self.assertTrue(np.isfinite(score))

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

    def test_rejection_confidence_study_cli_accepts_output_dir(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1] / "scripts" / "rejection_confidence_study.py"
        )
        spec = importlib.util.spec_from_file_location("rejection_confidence_study_cli", script_path)

        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        args = module.parse_args(["--output-dir", "out"])

        self.assertEqual(args.output_dir, Path("out"))

    def test_mixed_zoo_retrieval_cli_help_mentions_output_dir(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1] / "scripts" / "mixed_zoo_retrieval_research.py"
        )
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--output-dir", result.stdout)

    def test_mixed_zoo_parse_args_accepts_benchmark_mode(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1] / "scripts" / "mixed_zoo_retrieval_research.py"
        )
        spec = importlib.util.spec_from_file_location("mixed_zoo_retrieval_research_cli", script_path)

        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        args = module.parse_args(["--benchmark-mode", "lean", "--output-dir", "out"])

        self.assertEqual(args.benchmark_mode, "lean")
        self.assertEqual(args.output_dir, Path("out"))

    def test_build_supported_slices_allows_provisional_labels_in_lean_mode(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1] / "scripts" / "mixed_zoo_retrieval_research.py"
        )
        spec = importlib.util.spec_from_file_location("mixed_zoo_retrieval_research_cli", script_path)

        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        manifest = pd.DataFrame(
            [
                {
                    "model_id": f"hetero-{i}",
                    "audited_base_family": "fam/x",
                    "label_status": "provisional",
                    "query_bucket": "supported_candidate",
                    "common_schema": False,
                }
                for i in range(5)
            ]
        )

        strict = module.build_supported_candidate_slices(
            manifest,
            min_family_size=5,
            benchmark_mode="strict",
        )
        lean = module.build_supported_candidate_slices(
            manifest,
            min_family_size=5,
            benchmark_mode="lean",
        )

        self.assertTrue(strict["heterogeneous_supported"].empty)
        self.assertEqual(len(lean["heterogeneous_supported"]), 5)

    def test_mixed_zoo_build_supported_split_frames_preserves_slice_labels(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1] / "scripts" / "mixed_zoo_retrieval_research.py"
        )
        spec = importlib.util.spec_from_file_location("mixed_zoo_retrieval_research_cli", script_path)

        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        homogeneous = pd.DataFrame(
            [
                {"model_id": f"h-{i}", "audited_base_family": "fam/h"}
                for i in range(5)
            ]
        )
        heterogeneous = pd.DataFrame(
            [
                {"model_id": f"x-{i}", "audited_base_family": "fam/x"}
                for i in range(5)
            ]
        )

        gallery_supported, queries_supported, calibration_families = module.build_supported_split_frames(
            {
                "homogeneous_supported": homogeneous,
                "heterogeneous_supported": heterogeneous,
            }
        )

        self.assertGreater(len(gallery_supported), 0)
        self.assertEqual(
            set(queries_supported["evaluation_slice"].tolist()),
            {"homogeneous_supported", "heterogeneous_supported"},
        )
        self.assertTrue(
            set(queries_supported["split_role"].tolist()).issubset(
                {"calibration_supported", "test_supported"}
            )
        )
        self.assertGreaterEqual(len(calibration_families), 1)
        self.assertLessEqual(len(calibration_families), 2)

    def test_mixed_zoo_prediction_bridge_gate_holds_without_heterogeneous_rows(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1] / "scripts" / "mixed_zoo_retrieval_research.py"
        )
        spec = importlib.util.spec_from_file_location("mixed_zoo_retrieval_research_cli", script_path)

        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            pd.DataFrame(
                [
                    {
                        "method": "full_module_trace",
                        "slice": "all",
                        "top1_accuracy": 0.9,
                        "mrr": 0.95,
                    }
                ]
            ).to_csv(out / "retrieval_metrics.csv", index=False)

            module.write_prediction_bridge_gate(out)

            gate = (out / "prediction_bridge_gate.md").read_text(encoding="utf-8")
            self.assertIn("`HOLD`", gate)
            self.assertIn("No heterogeneous-supported retrieval rows were produced.", gate)

    def test_mixed_zoo_prediction_bridge_gate_goes_when_heterogeneous_signal_is_strong(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1] / "scripts" / "mixed_zoo_retrieval_research.py"
        )
        spec = importlib.util.spec_from_file_location("mixed_zoo_retrieval_research_cli", script_path)

        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            pd.DataFrame(
                [
                    {
                        "method": "full_module_trace",
                        "slice": "heterogeneous_supported",
                        "num_queries": 3,
                        "top1_accuracy": 0.60,
                        "mrr": 0.70,
                    }
                ]
            ).to_csv(out / "retrieval_metrics.csv", index=False)

            module.write_prediction_bridge_gate(out)

            gate = (out / "prediction_bridge_gate.md").read_text(encoding="utf-8")
            self.assertIn("`GO`", gate)
            self.assertIn("Mixed-zoo spectral neighborhoods are strong enough", gate)

    def test_mixed_zoo_prediction_bridge_gate_holds_when_heterogeneous_row_has_no_queries(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1] / "scripts" / "mixed_zoo_retrieval_research.py"
        )
        spec = importlib.util.spec_from_file_location("mixed_zoo_retrieval_research_cli", script_path)

        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            pd.DataFrame(
                [
                    {
                        "method": "full_module_trace",
                        "slice": "heterogeneous_supported",
                        "num_queries": 0,
                        "top1_accuracy": np.nan,
                        "mrr": np.nan,
                    }
                ]
            ).to_csv(out / "retrieval_metrics.csv", index=False)

            module.write_prediction_bridge_gate(out)

            gate = (out / "prediction_bridge_gate.md").read_text(encoding="utf-8")
            self.assertIn("`HOLD`", gate)
            self.assertIn("No heterogeneous-supported retrieval rows were produced.", gate)


if __name__ == "__main__":
    unittest.main()
