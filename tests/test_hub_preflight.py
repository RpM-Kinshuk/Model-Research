from unittest.mock import Mock

from fetch_data.hub_preflight import (
    ProbeResult,
    classify_fetchability_from_files,
    probe_repo_fetchability,
)


def test_classify_fetchability_from_files_flags_adapter_without_base_resolution():
    result = classify_fetchability_from_files(
        repo_id="owner/adapter-model",
        revision=None,
        files=["adapter_config.json", "adapter_model.safetensors"],
        loader_scenario="adapter_requires_base",
        base_model_name_or_path=None,
    )
    assert result.fetchability_status == "public_but_structurally_incomplete"
    assert result.failure_class == "adapter_base_unresolved"


def test_classify_fetchability_from_files_accepts_standard_transformers_repo():
    result = classify_fetchability_from_files(
        repo_id="owner/base-model",
        revision=None,
        files=["config.json", "model.safetensors", "tokenizer.json"],
        loader_scenario="standard_transformers",
        base_model_name_or_path=None,
    )
    assert result.fetchability_status == "public_and_likely_fetchable"
    assert result.failure_class is None


def test_probe_repo_fetchability_uses_api_without_downloading_weights():
    api = Mock()
    api.model_info.return_value = Mock(
        private=False,
        gated=False,
        config={"model_type": "llama"},
    )
    api.list_repo_files.return_value = ["config.json", "model.safetensors"]
    result = probe_repo_fetchability(
        api=api,
        repo_id="owner/model",
        revision=None,
        loader_scenario="standard_transformers",
        base_model_name_or_path=None,
        dry_run_probe=None,
    )
    assert isinstance(result, ProbeResult)
    assert result.availability_status == "public"
    assert result.fetchability_status == "public_and_likely_fetchable"
    api.model_info.assert_called_once()
    api.list_repo_files.assert_called_once()
