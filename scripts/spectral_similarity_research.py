from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_research.pipeline import main as pipeline_main, parse_args as pipeline_parse_args


def parse_args(argv: list[str] | None = None):
    args = pipeline_parse_args(argv)
    if args.output_dir == (
        REPO_ROOT
        / "research_runs"
        / "preliminary"
        / "retrieval"
        / "spectral_family_retrieval_strict_headline_v2"
    ):
        args.output_dir = REPO_ROOT / "research_runs" / "legacy" / "spectral_similarity_research"
    return args


def main(argv: list[str] | None = None) -> int:
    output_dir_flag = "--output-dir"
    has_explicit_output_dir = argv is not None and any(
        arg == output_dir_flag or arg.startswith(f"{output_dir_flag}=") for arg in argv
    )
    if has_explicit_output_dir:
        return pipeline_main(argv)
    return pipeline_main(
        (argv or [])
        + [
            "--output-dir",
            str(REPO_ROOT / "research_runs" / "legacy" / "spectral_similarity_research"),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
