from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

try:
    from huggingface_hub.errors import HfHubHTTPError
except ModuleNotFoundError:  # pragma: no cover - dependency may be unavailable in unit-test env
    class HfHubHTTPError(Exception):
        """Fallback type used when huggingface_hub is unavailable at import time."""


@dataclass
class ProbeResult:
    repo_id: str
    revision: str | None
    availability_status: str
    fetchability_status: str
    failure_class: str | None
    loader_scenario: str
    config_model_type: str | None
    base_model_name_or_path: str | None
    file_count: int


def classify_fetchability_from_files(
    *,
    repo_id: str,
    revision: str | None,
    files: list[str],
    loader_scenario: str,
    base_model_name_or_path: str | None,
) -> ProbeResult:
    files_lower = {file_name.lower() for file_name in files}
    if loader_scenario == "adapter_requires_base" and not base_model_name_or_path:
        return ProbeResult(
            repo_id=repo_id,
            revision=revision,
            availability_status="public",
            fetchability_status="public_but_structurally_incomplete",
            failure_class="adapter_base_unresolved",
            loader_scenario=loader_scenario,
            config_model_type=None,
            base_model_name_or_path=base_model_name_or_path,
            file_count=len(files_lower),
        )
    if loader_scenario == "standard_transformers" and "config.json" not in files_lower:
        return ProbeResult(
            repo_id=repo_id,
            revision=revision,
            availability_status="public",
            fetchability_status="public_but_structurally_incomplete",
            failure_class="config_missing",
            loader_scenario=loader_scenario,
            config_model_type=None,
            base_model_name_or_path=base_model_name_or_path,
            file_count=len(files_lower),
        )
    return ProbeResult(
        repo_id=repo_id,
        revision=revision,
        availability_status="public",
        fetchability_status="public_and_likely_fetchable",
        failure_class=None,
        loader_scenario=loader_scenario,
        config_model_type=None,
        base_model_name_or_path=base_model_name_or_path,
        file_count=len(files_lower),
    )


def probe_repo_fetchability(
    *,
    api,
    repo_id: str,
    revision: str | None,
    loader_scenario: str,
    base_model_name_or_path: str | None,
    dry_run_probe: Callable[[str, str | None], object] | None = None,
) -> ProbeResult:
    try:
        info = api.model_info(repo_id=repo_id, revision=revision, files_metadata=False)
        files = api.list_repo_files(repo_id=repo_id, revision=revision, repo_type="model")
    except HfHubHTTPError as error:
        status_code = getattr(getattr(error, "response", None), "status_code", None)
        if status_code in {401, 403}:
            return ProbeResult(
                repo_id, revision, "gated", "gated", "gated", loader_scenario, None, base_model_name_or_path, 0
            )
        if status_code == 404:
            return ProbeResult(
                repo_id,
                revision,
                "private_or_deleted",
                "private_or_deleted",
                "private_or_deleted",
                loader_scenario,
                None,
                base_model_name_or_path,
                0,
            )
        return ProbeResult(
            repo_id,
            revision,
            "probe_failed",
            "public_but_probe_failed",
            f"http_{status_code}",
            loader_scenario,
            None,
            base_model_name_or_path,
            0,
        )

    result = classify_fetchability_from_files(
        repo_id=repo_id,
        revision=revision,
        files=files,
        loader_scenario=loader_scenario,
        base_model_name_or_path=base_model_name_or_path,
    )
    result.config_model_type = None if not isinstance(getattr(info, "config", None), dict) else info.config.get("model_type")
    if dry_run_probe is not None and result.fetchability_status == "public_and_likely_fetchable":
        dry_run_probe(repo_id, revision)
    return result


def dry_run_small_file(repo_id: str, revision: str | None) -> object:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        revision=revision,
        dry_run=True,
    )
