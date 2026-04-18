from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from spectral_research.benchmark import build_audit_manifest


class AuditManifestTests(unittest.TestCase):
    @staticmethod
    def _manifest_row(
        model_id: str,
        family: str | None,
        *,
        label_status: str = "confirmed",
        query_bucket: str = "supported_candidate",
        common_schema: bool = True,
    ) -> dict[str, object]:
        return {
            "model_id": model_id,
            "audited_base_family": family,
            "label_status": label_status,
            "query_bucket": query_bucket,
            "common_schema": common_schema,
        }

    @staticmethod
    def _leaderboard_signature(row: pd.Series) -> tuple[object, object, object]:
        values = []
        for column in ["#Params (B)", "Generation", "Type"]:
            value = row[column]
            values.append(None if pd.isna(value) else value)
        return tuple(values)

    def test_leaderboard_duplicate_rows_use_stable_most_complete_entry(self) -> None:
        records = pd.DataFrame(
            [{"model_id": "child-a", "common_schema": True, "schema_name": "common_llm_7"}]
        )
        metadata = pd.DataFrame(
            [{"model_id": "child-a", "source_model": "base-a", "base_model_relation": "finetune"}]
        )
        overrides = pd.DataFrame(
            columns=["model_id", "audited_base_family", "label_status", "bucket_override"]
        )
        sparse_row = {
            "model_id": "child-a",
            "Architecture": "LlamaForCausalLM",
            "#Params (B)": pd.NA,
            "Generation": pd.NA,
            "Type": pd.NA,
        }
        complete_row = {
            "model_id": "child-a",
            "Architecture": "LlamaForCausalLM",
            "#Params (B)": 7.1,
            "Generation": 2,
            "Type": "chat",
        }

        manifest_a = build_audit_manifest(
            records,
            metadata,
            pd.DataFrame([sparse_row, complete_row]),
            overrides,
        )
        manifest_b = build_audit_manifest(
            records,
            metadata,
            pd.DataFrame([complete_row, sparse_row]),
            overrides,
        )

        row_a = manifest_a.iloc[0]
        row_b = manifest_b.iloc[0]

        self.assertEqual(self._leaderboard_signature(row_a), (7.1, 2, "chat"))
        self.assertEqual(self._leaderboard_signature(row_b), (7.1, 2, "chat"))

    def test_override_beats_candidate_family(self) -> None:
        records = pd.DataFrame(
            [{"model_id": "child-a", "common_schema": True, "schema_name": "common_llm_7"}]
        )
        metadata = pd.DataFrame(
            [{"model_id": "child-a", "source_model": "base-a", "base_model_relation": "finetune"}]
        )
        leaderboard = pd.DataFrame(
            [
                {
                    "model_id": "child-a",
                    "Architecture": "LlamaForCausalLM",
                    "#Params (B)": 7.1,
                    "Generation": 2,
                    "Type": "chat",
                }
            ]
        )
        overrides = pd.DataFrame(
            [
                {
                    "model_id": "child-a",
                    "audited_base_family": "base-a-v2",
                    "label_status": "confirmed",
                    "bucket_override": "",
                }
            ]
        )

        manifest = build_audit_manifest(records, metadata, leaderboard, overrides)
        row = manifest.iloc[0]

        self.assertEqual(row["candidate_family"], "base-a")
        self.assertEqual(row["audited_base_family"], "base-a-v2")
        self.assertEqual(row["label_status"], "confirmed")

    def test_bucket_assignment_marks_special_and_unknown_rows(self) -> None:
        records = pd.DataFrame(
            [
                {"model_id": "ft-a", "common_schema": True, "schema_name": "common_llm_7"},
                {"model_id": "merge-a", "common_schema": True, "schema_name": "common_llm_7"},
                {"model_id": "mystery", "common_schema": False, "schema_name": "weird"},
            ]
        )
        metadata = pd.DataFrame(
            [
                {"model_id": "ft-a", "source_model": "base-a", "base_model_relation": "finetune"},
                {"model_id": "merge-a", "source_model": "base-a_other", "base_model_relation": "merge"},
                {"model_id": "mystery", "source_model": "unknown", "base_model_relation": "unknown"},
            ]
        )
        leaderboard = pd.DataFrame(
            [
                {
                    "model_id": "ft-a",
                    "Architecture": "LlamaForCausalLM",
                    "#Params (B)": 7.0,
                    "Generation": 2,
                    "Type": "chat",
                },
                {
                    "model_id": "merge-a",
                    "Architecture": "LlamaForCausalLM",
                    "#Params (B)": 7.0,
                    "Generation": 2,
                    "Type": "merge",
                },
                {
                    "model_id": "mystery",
                    "Architecture": "Other",
                    "#Params (B)": 3.0,
                    "Generation": 1,
                    "Type": "base",
                },
            ]
        )
        overrides = pd.DataFrame(
            columns=["model_id", "audited_base_family", "label_status", "bucket_override"]
        )

        manifest = build_audit_manifest(records, metadata, leaderboard, overrides).set_index(
            "model_id"
        )

        self.assertEqual(manifest.loc["ft-a", "query_bucket"], "supported_candidate")
        self.assertEqual(manifest.loc["merge-a", "query_bucket"], "unsupported_special")
        self.assertEqual(manifest.loc["mystery", "query_bucket"], "unsupported_unknown")

    def test_bucket_override_takes_precedence_when_present(self) -> None:
        records = pd.DataFrame(
            [{"model_id": "merge-a", "common_schema": True, "schema_name": "common_llm_7"}]
        )
        metadata = pd.DataFrame(
            [{"model_id": "merge-a", "source_model": "base-a_other", "base_model_relation": "merge"}]
        )
        leaderboard = pd.DataFrame(
            [
                {
                    "model_id": "merge-a",
                    "Architecture": "LlamaForCausalLM",
                    "#Params (B)": 7.0,
                    "Generation": 2,
                    "Type": "merge",
                }
            ]
        )
        overrides = pd.DataFrame(
            [
                {
                    "model_id": "merge-a",
                    "audited_base_family": "base-a",
                    "label_status": "confirmed",
                    "bucket_override": "supported_candidate",
                }
            ]
        )

        manifest = build_audit_manifest(records, metadata, leaderboard, overrides).set_index(
            "model_id"
        )

        self.assertEqual(manifest.loc["merge-a", "query_bucket"], "supported_candidate")

    def test_source_relation_and_schema_drive_fallback_labels_and_size_buckets(self) -> None:
        records = pd.DataFrame(
            [
                {"model_id": "base-anchor", "common_schema": True, "schema_name": "common_llm_7"},
                {"model_id": "hetero-known", "common_schema": False, "schema_name": "attn_q + mlp_up"},
                {"model_id": "no-family", "common_schema": True, "schema_name": "common_llm_7"},
            ]
        )
        metadata = pd.DataFrame(
            [
                {
                    "model_id": "base-anchor",
                    "source_model": "unknown",
                    "base_model_relation": "source",
                },
                {
                    "model_id": "hetero-known",
                    "source_model": "base-b",
                    "base_model_relation": "finetune",
                },
                {
                    "model_id": "no-family",
                    "source_model": "unknown",
                    "base_model_relation": "unknown",
                },
            ]
        )
        leaderboard = pd.DataFrame(
            [
                {
                    "model_id": "base-anchor",
                    "Architecture": "LlamaForCausalLM",
                    "#Params (B)": 3.0,
                    "Generation": 1,
                    "Type": "base",
                },
                {
                    "model_id": "hetero-known",
                    "Architecture": "LlamaForCausalLM",
                    "#Params (B)": 13.0,
                    "Generation": 2,
                    "Type": "chat",
                },
                {
                    "model_id": "no-family",
                    "Architecture": "Other",
                    "#Params (B)": 2.9,
                    "Generation": 1,
                    "Type": "base",
                },
            ]
        )
        overrides = pd.DataFrame(
            columns=["model_id", "audited_base_family", "label_status", "bucket_override"]
        )

        manifest = build_audit_manifest(records, metadata, leaderboard, overrides).set_index(
            "model_id"
        )

        self.assertEqual(manifest.loc["base-anchor", "candidate_family"], "base-anchor")
        self.assertEqual(manifest.loc["base-anchor", "audited_base_family"], "base-anchor")
        self.assertEqual(manifest.loc["base-anchor", "label_status"], "provisional")
        self.assertEqual(manifest.loc["base-anchor", "size_bucket"], "small")
        self.assertEqual(manifest.loc["hetero-known", "query_bucket"], "heterogeneous")
        self.assertEqual(manifest.loc["hetero-known", "size_bucket"], "medium")
        self.assertTrue(pd.isna(manifest.loc["no-family", "audited_base_family"]))
        self.assertEqual(manifest.loc["no-family", "label_status"], "unknown")
        self.assertEqual(manifest.loc["no-family", "size_bucket"], "small")

    def test_bucket_override_takes_precedence_over_default_assignment(self) -> None:
        records = pd.DataFrame(
            [{"model_id": "ft-a", "common_schema": True, "schema_name": "common_llm_7"}]
        )
        metadata = pd.DataFrame(
            [{"model_id": "ft-a", "source_model": "base-a", "base_model_relation": "finetune"}]
        )
        leaderboard = pd.DataFrame(
            [
                {
                    "model_id": "ft-a",
                    "Architecture": "LlamaForCausalLM",
                    "#Params (B)": 7.0,
                    "Generation": 2,
                    "Type": "chat",
                }
            ]
        )
        overrides = pd.DataFrame(
            [
                {
                    "model_id": "ft-a",
                    "audited_base_family": "base-a",
                    "label_status": "confirmed",
                    "bucket_override": "unsupported_special",
                }
            ]
        )

        manifest = build_audit_manifest(records, metadata, leaderboard, overrides).set_index(
            "model_id"
        )

        self.assertEqual(manifest.loc["ft-a", "query_bucket"], "unsupported_special")

    def test_duplicate_leaderboard_rows_collapse_deterministically(self) -> None:
        records = pd.DataFrame(
            [{"model_id": "dup-a", "common_schema": True, "schema_name": "common_llm_7"}]
        )
        metadata = pd.DataFrame(
            [{"model_id": "dup-a", "source_model": "base-a", "base_model_relation": "finetune"}]
        )
        leaderboard_a = pd.DataFrame(
            [
                {
                    "model_id": "dup-a",
                    "Architecture": "Unknown",
                    "#Params (B)": 2.9,
                    "Generation": 1,
                    "Type": "chat",
                },
                {
                    "model_id": "dup-a",
                    "Architecture": "LlamaForCausalLM",
                    "#Params (B)": 13.0,
                    "Generation": 2,
                    "Type": "chat",
                },
            ]
        )
        leaderboard_b = leaderboard_a.iloc[::-1].reset_index(drop=True)
        overrides = pd.DataFrame(
            columns=["model_id", "audited_base_family", "label_status", "bucket_override"]
        )

        manifest_a = build_audit_manifest(records, metadata, leaderboard_a, overrides).set_index(
            "model_id"
        )
        manifest_b = build_audit_manifest(records, metadata, leaderboard_b, overrides).set_index(
            "model_id"
        )

        self.assertEqual(manifest_a.loc["dup-a", "Architecture"], "LlamaForCausalLM")
        self.assertEqual(manifest_a.loc["dup-a", "Architecture"], manifest_b.loc["dup-a", "Architecture"])
        self.assertEqual(manifest_a.loc["dup-a", "size_bucket"], "medium")
        self.assertEqual(manifest_a.loc["dup-a", "size_bucket"], manifest_b.loc["dup-a", "size_bucket"])

    def test_repo_ids_join_after_slash_to_dash_normalization(self) -> None:
        records = pd.DataFrame(
            [
                {
                    "model_id": "meta-llama--Llama-2-7b",
                    "display_name": "meta-llama/Llama-2-7b",
                    "common_schema": True,
                    "schema_name": "common_llm_7",
                }
            ]
        )
        metadata = pd.DataFrame(
            [
                {
                    "model_id": "meta-llama/Llama-2-7b",
                    "source_model": "unknown",
                    "base_model_relation": "source",
                }
            ]
        )
        leaderboard = pd.DataFrame(
            [
                {
                    "model_id": "meta-llama/Llama-2-7b",
                    "Architecture": "LlamaForCausalLM",
                    "#Params (B)": 7.0,
                    "Generation": 1,
                    "Type": "base",
                }
            ]
        )
        overrides = pd.DataFrame(
            [
                {
                    "model_id": "meta-llama/Llama-2-7b",
                    "audited_base_family": "meta-llama/Llama-2-7b",
                    "label_status": "confirmed",
                    "bucket_override": "",
                }
            ]
        )

        manifest = build_audit_manifest(records, metadata, leaderboard, overrides).set_index(
            "model_id"
        )

        self.assertEqual(
            manifest.loc["meta-llama--Llama-2-7b", "audited_base_family"],
            "meta-llama/Llama-2-7b",
        )
        self.assertEqual(
            manifest.loc["meta-llama--Llama-2-7b", "Architecture"],
            "LlamaForCausalLM",
        )
        self.assertEqual(
            manifest.loc["meta-llama--Llama-2-7b", "label_status"],
            "confirmed",
        )

    def test_split_artifacts_partition_supported_families_deterministically(self) -> None:
        from spectral_research.benchmark import build_split_artifacts

        families = [f"fam-{suffix}" for suffix in "abcdef"]
        manifest = pd.DataFrame(
            [
                self._manifest_row(f"{family}-{index}", family)
                for family in families
                for index in range(5)
            ]
        )

        artifacts = build_split_artifacts(
            manifest,
            min_family_size=5,
            calibration_fraction=0.5,
            seed=7,
        )

        self.assertEqual(
            set(artifacts),
            {
                "gallery_supported",
                "queries_supported",
                "queries_unsupported",
                "queries_heterogeneous",
            },
        )

        gallery = artifacts["gallery_supported"]
        queries = artifacts["queries_supported"]
        self.assertEqual(set(gallery["audited_base_family"]), set(families))
        self.assertEqual(set(queries["audited_base_family"]), set(families))
        self.assertTrue(artifacts["queries_unsupported"].empty)
        self.assertTrue(artifacts["queries_heterogeneous"].empty)

        calibration_families = set(
            np.random.default_rng(7).permutation(sorted(families))[:3].tolist()
        )
        for family in families:
            gallery_ids = gallery.loc[
                gallery["audited_base_family"].eq(family), "model_id"
            ].tolist()
            query_ids = queries.loc[
                queries["audited_base_family"].eq(family), "model_id"
            ].tolist()
            expected_ids = [f"{family}-{index}" for index in range(5)]

            self.assertEqual(gallery_ids, expected_ids[:4])
            self.assertEqual(query_ids, expected_ids[4:])

            split_role = queries.loc[
                queries["audited_base_family"].eq(family), "split_role"
            ].iloc[0]
            expected_role = (
                "calibration_supported"
                if family in calibration_families
                else "test_supported"
            )
            self.assertEqual(split_role, expected_role)

    def test_split_artifacts_relax_family_size_and_route_non_primary_rows(self) -> None:
        from spectral_research.benchmark import build_split_artifacts

        kept_families = [f"fam-{suffix}" for suffix in "abcde"]
        manifest_rows = [
            self._manifest_row(f"{family}-{index}", family)
            for family in kept_families
            for index in range(5)
        ]
        manifest_rows.extend(
            [
                self._manifest_row("small-0", "fam-small"),
                self._manifest_row("small-1", "fam-small"),
                self._manifest_row("small-2", "fam-small"),
                self._manifest_row("small-3", "fam-small"),
                self._manifest_row(
                    "unknown-0",
                    "fam-unknown",
                    query_bucket="unsupported_unknown",
                ),
                self._manifest_row(
                    "special-0",
                    "fam-special",
                    query_bucket="unsupported_special",
                ),
                self._manifest_row(
                    "hetero-0",
                    "fam-a",
                    query_bucket="heterogeneous",
                    common_schema=False,
                ),
                self._manifest_row(
                    "hetero-unknown",
                    "fam-z",
                    label_status="unknown",
                    query_bucket="heterogeneous",
                    common_schema=False,
                ),
            ]
        )
        manifest = pd.DataFrame(manifest_rows)

        artifacts = build_split_artifacts(
            manifest,
            min_family_size=6,
            calibration_fraction=0.4,
            seed=3,
        )

        gallery = artifacts["gallery_supported"]
        queries = artifacts["queries_supported"]
        unsupported = artifacts["queries_unsupported"]
        heterogeneous = artifacts["queries_heterogeneous"]

        self.assertEqual(set(gallery["audited_base_family"]), set(kept_families))
        self.assertEqual(set(queries["audited_base_family"]), set(kept_families))
        self.assertEqual(len(gallery), 20)
        self.assertEqual(len(queries), 5)
        self.assertEqual(set(unsupported["split_role"]), {"unsupported_query"})
        self.assertEqual(set(heterogeneous["split_role"]), {"heterogeneous_query"})
        self.assertEqual(
            set(unsupported["model_id"]),
            {"small-0", "small-1", "small-2", "small-3", "unknown-0", "special-0"},
        )
        self.assertEqual(heterogeneous["model_id"].tolist(), ["hetero-0"])


if __name__ == "__main__":
    unittest.main()
