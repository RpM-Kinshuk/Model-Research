from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


ARTIFACT_MANIFEST_FILE = "artifact_manifest.json"
ARTIFACT_MANIFEST_SCHEMA = "htsr-artifact-manifest-v1"


class ArtifactValidationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_shape(path: Path) -> tuple[int, list[str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            columns = next(reader)
        except StopIteration:
            return 0, []
        rows = sum(1 for _ in reader)
    return rows, columns


def write_artifact_manifest(run_dir: str | Path, *, run_id: str) -> Path:
    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"artifact directory does not exist: {root}")
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name == ARTIFACT_MANIFEST_FILE:
            continue
        record: dict[str, Any] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        if path.suffix.lower() == ".csv":
            rows, columns = _csv_shape(path)
            record.update({"rows": rows, "columns": columns})
        artifacts[path.name] = record

    destination = root / ARTIFACT_MANIFEST_FILE
    destination.write_text(
        json.dumps(
            {
                "schema_version": ARTIFACT_MANIFEST_SCHEMA,
                "run_id": str(run_id),
                "artifacts": artifacts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def validate_artifact_manifest(
    run_dir: str | Path,
    *,
    required_files: Sequence[str],
    expected_run_id: str | None = None,
    validate_all: bool = False,
) -> dict[str, Any]:
    root = Path(run_dir)
    manifest_path = root / ARTIFACT_MANIFEST_FILE
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(
            f"invalid {ARTIFACT_MANIFEST_FILE}: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise ArtifactValidationError(f"{ARTIFACT_MANIFEST_FILE} must be an object")
    if manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA:
        raise ArtifactValidationError(
            f"unsupported {ARTIFACT_MANIFEST_FILE} schema_version"
        )
    if expected_run_id is not None and manifest.get("run_id") != expected_run_id:
        raise ArtifactValidationError(
            f"artifact manifest run_id does not match {expected_run_id!r}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ArtifactValidationError("artifact manifest artifacts must be an object")

    required = tuple(dict.fromkeys(str(name) for name in required_files))
    missing_records = [name for name in required if name not in artifacts]
    if missing_records:
        raise ArtifactValidationError(
            f"artifact manifest is missing required records: {missing_records}"
        )
    selected = tuple(artifacts) if validate_all else required
    for name in selected:
        if Path(name).name != name:
            raise ArtifactValidationError(f"invalid artifact filename: {name!r}")
        record = artifacts.get(name)
        if not isinstance(record, dict):
            raise ArtifactValidationError(f"invalid artifact record: {name}")
        path = root / name
        if not path.is_file():
            raise ArtifactValidationError(f"artifact file is missing: {name}")
        observed_size = path.stat().st_size
        if record.get("size_bytes") != observed_size:
            raise ArtifactValidationError(
                f"artifact size mismatch for {name}: "
                f"recorded={record.get('size_bytes')!r}, observed={observed_size}"
            )
        observed_digest = _sha256(path)
        if record.get("sha256") != observed_digest:
            raise ArtifactValidationError(f"artifact digest mismatch for {name}")
    return manifest


__all__ = [
    "ARTIFACT_MANIFEST_FILE",
    "ARTIFACT_MANIFEST_SCHEMA",
    "ArtifactValidationError",
    "validate_artifact_manifest",
    "write_artifact_manifest",
]
