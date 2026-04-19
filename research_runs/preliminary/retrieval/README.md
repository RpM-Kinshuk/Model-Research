# Retrieval Runs

This group contains both canonical retrieval artifacts and support or intermediate runs.

## Canonical artifacts

These are the retrieval outputs that should be treated as the primary preserved surfaces:

- `spectral_family_retrieval_strict_headline_v2`
- `lean_mixed_zoo_benchmark`

## Support and intermediate runs

These are preserved for context, comparison, or reruns, but they are not the canonical retrieval artifacts:

- `mixed_zoo_retrieval_cycle1`
- `spectral_family_retrieval`
- `spectral_family_retrieval_curated_v1`
- `spectral_family_retrieval_root_base_only_v1`
- `spectral_family_retrieval_strict_metadata_backfill_v1`

Current preliminary work should prefer the canonical artifacts above unless a helper explicitly depends on one of the support or intermediate runs.
