#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from spectral_research.study.contracts import StudyInputs
from spectral_research.study.protocol import StudyProtocol, load_protocol
from spectral_research.study.runner import run_study


_SETTING_OVERRIDES = (
    ("schema_version", "--schema-version"),
    ("random_seed", "--random-seed"),
    ("bootstrap_replicates", "--bootstrap-replicates"),
    ("signflip_replicates", "--signflip-replicates"),
    ("outer_folds", "--outer-folds"),
    ("inner_folds", "--inner-folds"),
    ("ridge_alphas", "--ridge-alphas"),
    ("minimum_components", "--minimum-components"),
    ("minimum_lineage_families", "--minimum-lineage-families"),
    ("min_feature_coverage", "--min-feature-coverage"),
    ("min_pair_coverage", "--min-pair-coverage"),
    ("min_role_coverage", "--min-role-coverage"),
    ("cluster_count_rule", "--cluster-count-rule"),
    ("holdout_repeats", "--holdout-repeats"),
    ("holdout_test_fraction", "--holdout-test-fraction"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the content-addressed HTSR lineage/performance study."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--development-stats", type=Path, required=True)
    parser.add_argument("--development-targets", type=Path, required=True)
    parser.add_argument("--external-stats", type=Path)
    parser.add_argument("--external-targets", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--schema-version", help=argparse.SUPPRESS)
    parser.add_argument("--random-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--bootstrap-replicates", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--signflip-replicates", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--outer-folds", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--inner-folds", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        metavar="ALPHA",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--minimum-components", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--minimum-lineage-families", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--min-feature-coverage", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--min-pair-coverage", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--min-role-coverage", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--cluster-count-rule", help=argparse.SUPPRESS)
    parser.add_argument("--holdout-repeats", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--holdout-test-fraction", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _reject_conflicting_overrides(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    protocol: StudyProtocol,
) -> None:
    for destination, option in _SETTING_OVERRIDES:
        override = getattr(args, destination)
        expected = getattr(protocol, destination)
        if destination == "ridge_alphas" and override is not None:
            override = tuple(override)
        if override is not None and override != expected:
            parser.error(
                f"{option} conflicts with protocol {protocol.protocol_version!r}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.external_stats is None) != (args.external_targets is None):
        parser.error("--external-stats and --external-targets must be supplied together")
    try:
        protocol = load_protocol(args.protocol)
    except (OSError, TypeError, ValueError) as error:
        parser.error(f"cannot load protocol {args.protocol}: {error}")
    _reject_conflicting_overrides(parser, args, protocol)

    inputs = StudyInputs(
        metadata=args.metadata,
        labels=args.labels,
        graph=args.graph,
        development_stats=args.development_stats,
        development_targets=args.development_targets,
        external_stats=args.external_stats,
        external_targets=args.external_targets,
        cache_root=args.cache_root,
        output_root=args.output_root,
    )
    result = run_study(inputs, protocol, dry_run=args.dry_run)
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "run_dir": str(result.run_dir) if result.run_dir is not None else None,
                "dry_run": result.dry_run,
                "cache_reused": result.cache_reused,
                "summary": dict(result.summary),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
