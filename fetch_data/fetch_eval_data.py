import re
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError


HF_LINK_PATTERN = re.compile(r"huggingface\.co/([^\"?#]+)")
KNOWN_PRECISIONS = {
    "bf16",
    "bfloat16",
    "float16",
    "float32",
    "fp16",
    "fp32",
    "int4",
    "int8",
    "4bit",
    "8bit",
}


HF_RATE_LIMIT_WINDOW_SECONDS = 300
HF_RATE_LIMIT_MAX_REQUESTS = 500
# Keep a safety margin for clock skew, retries, and incidental requests.
HF_RATE_LIMIT_BUFFER_REQUESTS = 40
HF_EFFECTIVE_REQUEST_BUDGET = HF_RATE_LIMIT_MAX_REQUESTS - HF_RATE_LIMIT_BUFFER_REQUESTS
HF_MIN_REQUEST_INTERVAL_SECONDS = 0.15
HF_RATE_LIMIT_MAX_RETRIES = 5
HF_RETRY_BACKOFF_BASE_SECONDS = 10


def parse_model_string(model_str: str) -> Tuple[str, Optional[str]]:
    """
    Match model-loader behavior for strings like "org/model@revision".
    Returns (repo_id, revision).
    """
    model_str = str(model_str).strip()
    if "@" in model_str and not model_str.startswith("@"):
        repo_id, revision = model_str.split("@", 1)
        return repo_id.strip(), revision.strip() or None
    return model_str, None


def normalize_repo_id(raw_value: object) -> Optional[str]:
    """Normalize model identifiers into canonical Hugging Face repo IDs."""
    if pd.isna(raw_value):
        return None

    value = str(raw_value).strip()
    if not value:
        return None

    # Extract repo from leaderboard HTML model cells.
    match = HF_LINK_PATTERN.search(value)
    if match:
        value = match.group(1).strip()

    value = re.sub(r"^https?://(?:www\.)?huggingface\.co/", "", value, flags=re.IGNORECASE)
    value = value.split("?", 1)[0].split("#", 1)[0].strip().strip("/")
    value = re.sub(r"<[^>]+>", "", value).strip()

    if not value:
        return None

    repo_id, _ = parse_model_string(value)
    repo_id = repo_id.strip().strip("/")

    # Fallback for escaped IDs like owner--model.
    if "/" not in repo_id and "--" in repo_id:
        owner, model_name = repo_id.split("--", 1)
        if owner and model_name:
            repo_id = f"{owner}/{model_name}"

    if "/" not in repo_id:
        return None

    return repo_id


def eval_name_to_repo_id(eval_name: object) -> Optional[str]:
    """
    Best-effort fallback for eval_name format like "org_model_precision".
    This is only used when richer columns are unavailable.
    """
    if pd.isna(eval_name):
        return None

    value = str(eval_name).strip()
    if not value:
        return None

    parts = value.split("_")
    if len(parts) < 2:
        return None

    if parts[-1].lower() in KNOWN_PRECISIONS:
        parts = parts[:-1]

    if len(parts) < 2:
        return None

    repo_from_eval = "_".join(parts[1:])
    return normalize_repo_id(f"{parts[0]}/{repo_from_eval}")


def build_leaderboard_repo_ids(df_perf: pd.DataFrame) -> pd.Series:
    """Build canonical repo IDs from leaderboard rows."""
    repo_ids = pd.Series([None] * len(df_perf), index=df_perf.index, dtype="object")

    if "Model" in df_perf.columns:
        repo_ids = df_perf["Model"].map(normalize_repo_id)

    if "Base Model" in df_perf.columns:
        base_ids = df_perf["Base Model"].map(normalize_repo_id)
        repo_ids = repo_ids.where(repo_ids.notna(), base_ids)

    if "eval_name" in df_perf.columns:
        eval_ids = df_perf["eval_name"].map(eval_name_to_repo_id)
        repo_ids = repo_ids.where(repo_ids.notna(), eval_ids)

    return repo_ids


def prune_old_requests(request_times: Deque[float], window_seconds: int) -> None:
    """Drop request timestamps that are outside the current rate-limit window."""
    now = time.monotonic()
    cutoff = now - window_seconds
    while request_times and request_times[0] <= cutoff:
        request_times.popleft()


def wait_for_request_budget(
    request_times: Deque[float], max_requests: int, window_seconds: int
) -> None:
    """Block until another request fits in the rolling rate-limit budget."""
    prune_old_requests(request_times, window_seconds)
    if len(request_times) < max_requests:
        return

    now = time.monotonic()
    wait_seconds = (request_times[0] + window_seconds) - now + 1.0
    wait_seconds = max(wait_seconds, 1.0)
    print(
        f"Rate-limit budget reached ({len(request_times)}/{max_requests} in "
        f"{window_seconds}s). Sleeping {wait_seconds:.1f}s..."
    )
    time.sleep(wait_seconds)
    prune_old_requests(request_times, window_seconds)


def is_rate_limit_error(error: Exception) -> bool:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code == 429:
        return True

    message = str(error).lower()
    return "rate limit" in message or "too many requests" in message or "429" in message


def retry_after_seconds(error: Exception) -> Optional[float]:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after is None:
        return None

    try:
        return max(float(retry_after), 0.0)
    except (TypeError, ValueError):
        return None


script_dir = Path(__file__).resolve().parent
atlas_path = script_dir / "sampled_metadata.csv"
output_file = script_dir / "sampled_leaderboard_merged.csv"

print("Loading local metadata...")
atlas_df = pd.read_csv(atlas_path)
if "full_name" not in atlas_df.columns:
    raise KeyError("atlas_metadata.csv is missing required column: full_name")

atlas_df["atlas_repo_id"] = atlas_df["full_name"].map(normalize_repo_id)
atlas_df["atlas_repo_key"] = atlas_df["atlas_repo_id"].str.casefold()
atlas_key_set = set(atlas_df["atlas_repo_key"].dropna())

print("Downloading Open LLM Leaderboard dataset...")
lb_dataset = load_dataset("open-llm-leaderboard/contents", split="train")
df_perf = lb_dataset.to_pandas()
df_perf["leaderboard_repo_id"] = build_leaderboard_repo_ids(df_perf)
df_perf["leaderboard_repo_key"] = df_perf["leaderboard_repo_id"].str.casefold()

print("Filtering leaderboard data for models in your atlas metadata...")
df_perf_filtered = df_perf[df_perf["leaderboard_repo_key"].isin(atlas_key_set)].copy()

matched_key_count = df_perf_filtered["leaderboard_repo_key"].dropna().nunique()
print(
    f"Found {len(df_perf_filtered)} leaderboard rows covering "
    f"{matched_key_count} out of {len(atlas_key_set)} atlas models."
)

if matched_key_count == 0:
    print("No model-name matches were found after normalization.")
    print("Sample atlas IDs:", atlas_df["atlas_repo_id"].dropna().head(10).tolist())
    print("Sample leaderboard IDs:", df_perf["leaderboard_repo_id"].dropna().head(10).tolist())

api = HfApi()
model_metadata = []
matched_model_ids = sorted(df_perf_filtered["leaderboard_repo_id"].dropna().unique())

print(f"Fetching inherent properties from HF Hub for {len(matched_model_ids)} models...")
request_times: Deque[float] = deque()
last_request_time = 0.0
for model_id in matched_model_ids:
    info = None
    for attempt in range(1, HF_RATE_LIMIT_MAX_RETRIES + 1):
        wait_for_request_budget(
            request_times=request_times,
            max_requests=HF_EFFECTIVE_REQUEST_BUDGET,
            window_seconds=HF_RATE_LIMIT_WINDOW_SECONDS,
        )

        elapsed = time.monotonic() - last_request_time
        if elapsed < HF_MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(HF_MIN_REQUEST_INTERVAL_SECONDS - elapsed)

        try:
            info = api.model_info(model_id)
            now = time.monotonic()
            request_times.append(now)
            last_request_time = now
            break
        except HfHubHTTPError as e:
            now = time.monotonic()
            request_times.append(now)
            last_request_time = now

            if is_rate_limit_error(e) and attempt < HF_RATE_LIMIT_MAX_RETRIES:
                server_wait = retry_after_seconds(e)
                local_backoff = HF_RETRY_BACKOFF_BASE_SECONDS * attempt
                sleep_seconds = (server_wait + 1.0) if server_wait is not None else local_backoff
                print(
                    f"[RATE LIMIT] {model_id} attempt {attempt}/{HF_RATE_LIMIT_MAX_RETRIES}; "
                    f"sleeping {sleep_seconds:.1f}s before retry..."
                )
                time.sleep(sleep_seconds)
                continue

            # Models can be deleted, private, or gated.
            print(f"[WARN] Could not fetch metadata for {model_id}: {e}")
            break
        except Exception as e:
            # Non-HTTP errors are logged and skipped.
            print(f"[WARN] Could not fetch metadata for {model_id}: {e}")
            break

    if info is None:
        continue

    params = None
    if info.safetensors and info.safetensors.parameters:
        params = info.safetensors.parameters.get("total")

    config = info.config if isinstance(info.config, dict) else {}
    model_metadata.append(
        {
            "leaderboard_repo_key": model_id.casefold(),
            "architecture": config.get("model_type", "unknown"),
            "parameter_count": params,
            "downloads": info.downloads,
            "likes": info.likes,
            "pipeline_tag": info.pipeline_tag,
        }
    )

df_meta = pd.DataFrame(model_metadata)
if not df_meta.empty:
    df_perf_enriched = pd.merge(df_perf_filtered, df_meta, on="leaderboard_repo_key", how="left")
else:
    df_perf_enriched = df_perf_filtered.copy()

print("Merging with original atlas_metadata columns...")
df_final = pd.merge(
    atlas_df,
    df_perf_enriched,
    left_on="atlas_repo_key",
    right_on="leaderboard_repo_key",
    how="inner",
    suffixes=("_atlas", "_lb"),
)

df_final.to_csv(output_file, index=False)

unmatched_count = (~atlas_df["atlas_repo_key"].isin(set(df_final["atlas_repo_key"].dropna()))).sum()
print(f"Successfully saved merged data to {output_file}")
print(f"Final merged rows: {len(df_final)}")
print(f"Unmatched atlas models: {int(unmatched_count)}")
