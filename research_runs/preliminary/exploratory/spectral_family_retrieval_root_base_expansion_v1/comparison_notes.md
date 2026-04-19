# Min-5 Expansion Comparison Notes

- Baseline run: `research_runs/preliminary/retrieval/spectral_family_retrieval_strict_headline_v2`
- Threshold-only sensitivity run: `research_runs/preliminary/exploratory/spectral_family_retrieval_strict_headline_min5_v1`
- Derived-freeze expansion run: `research_runs/preliminary/exploratory/spectral_family_retrieval_root_base_expansion_v1`
- Decision question: does relaxing the support bar and promoting a few near-miss official base families add useful benchmark coverage without materially weakening retrieval quality?

## Record
- Baseline confirmed families: `12`
- Expansion confirmed families: `15`
- Baseline confirmed rows: `225`
- Expansion confirmed rows: `241`
- Baseline gallery / supported queries: `153 / 72`
- Expansion gallery / supported queries: `165 / 76`
- Newly admitted families:
  - `mistralai/Mistral-7B-v0.3`
  - `mistralai/Mistral-Small-24B-Base-2501`
  - `Qwen/Qwen2.5-1.5B`
- `full_module_trace` all-slice delta:
  - top-1 `0.918 -> 0.881`
  - MRR `0.950 -> 0.914`
- `attention_trace` all-slice delta:
  - top-1 `0.918 -> 0.881`
  - MRR `0.946 -> 0.914`
- `pl_alpha_scalar` all-slice delta:
  - top-1 `0.857 -> 0.786`
  - MRR `0.903 -> 0.852`
- Keep or reject the expansion as the new headline benchmark: `reject`

## Why

The threshold-only `min_family_size=5` run changed nothing, so the family cutoff alone was not the real bottleneck.

The derived-freeze expansion did add three plausible official base families and increased the supported pool modestly, but the main retrieval quality dropped across all core spectral methods. That makes this useful as an exploratory artifact, not a better headline benchmark.

The practical conclusion is:

- keep `strict_headline_v2` as the main preliminary benchmark
- keep `root_base_expansion_v1` as a sensitivity artifact
- treat metadata cleanup on the confirmed strict slice as the higher-value next step than lowering the support bar
