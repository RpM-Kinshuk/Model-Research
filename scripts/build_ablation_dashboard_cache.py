from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alpha_cluster_dashboard.ablation import DEFAULT_ABLATION_METRICS_DIR
from alpha_cluster_dashboard.dashboard_cache import (
    DEFAULT_ABLATION_METADATA,
    DEFAULT_CACHE_DIR,
    DashboardCacheConfig,
    load_or_build_dashboard_cache,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the persistent model-zoo spectral dashboard cache.")
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_ABLATION_METRICS_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_ABLATION_METADATA)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--n-bins", type=int, default=24)
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--n-neighbors", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = DashboardCacheConfig(
        n_bins=args.n_bins,
        n_clusters=args.n_clusters,
        n_neighbors=args.n_neighbors,
        random_state=args.random_state,
    )

    def progress(fraction: float, message: str) -> None:
        print(f"{fraction * 100:5.1f}% {message}", flush=True)

    print(f"Metrics: {args.metrics_dir}")
    print(f"Metadata: {args.metadata_csv}")
    print(f"Cache: {args.cache_dir}")
    print(f"Config: {config}")
    cache = load_or_build_dashboard_cache(
        args.metrics_dir,
        args.metadata_csv,
        args.cache_dir,
        config=config,
        force=args.force,
        progress=progress,
    )
    print(f"Models: {len(cache.metadata):,}")
    print(f"Cache: {cache.cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
