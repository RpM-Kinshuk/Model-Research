# Strict Metadata Backfill Comparison Notes

- Baseline run: `research_runs/spectral_family_retrieval_strict_headline_v2`
- Backfill run: `research_runs/spectral_family_retrieval_strict_metadata_backfill_v1`
- Backfill source: `research_runs/spectral_family_retrieval/strict_metadata_backfill_leaderboard_v1.csv`

## What Changed

- Confirmed rows stayed the same: `225`
- Confirmed rows with architecture metadata changed from `41` to `225`
- Confirmed rows missing architecture metadata changed from `184` to `0`
- The same family set and same strict freeze were used

## Retrieval Deltas

- `full_module_trace`
  - all-slice unchanged at top-1 `0.918`, MRR `0.950`
  - architecture-matched query count expanded from `17` to `49`
- `attention_trace`
  - all-slice unchanged at top-1 `0.918`, MRR `0.946`
  - architecture-matched query count expanded from `17` to `49`
- `metadata_only`
  - all-slice jumped from top-1 `0.184`, MRR `0.323`
  - to top-1 `0.918`, MRR `0.959`

## Interpretation

The metadata cleanup worked mechanically:

- matched slices now cover the full supported query set instead of a small metadata-present subset
- spectral retrieval quality stayed stable

But the cleanup also exposed a benchmark limitation:

- on this strict root/base benchmark, canonical architecture plus parameter size already identify the family extremely well
- after metadata backfill, `metadata_only` becomes a very strong control rather than a weak baseline

So this run is useful, but it changes the interpretation:

- the benchmark is now better documented
- the spectral methods still work well
- however, the claim that spectral fingerprints clearly beat metadata no longer holds on the metadata-complete version of this narrow benchmark
- the next evaluation step should emphasize stronger metadata-controlled comparisons rather than more metadata cleanup
