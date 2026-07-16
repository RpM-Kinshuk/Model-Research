# Adversarial Review Notes

Three independent review agents critiqued the corrected study after GPT-2/GPT-NeoX/OPT role canonicalization and the structural baseline were added.

## Consensus

- Do not headline spectral family recovery: `structural_basics` dominates coarse-family retrieval.
- Keep `paper_spearman` as the paper-faithful diagnostic baseline, not the operational default.
- Carry forward `hybrid_rank` and `rank_rbf` for performance-neighborhood prediction, where they beat the structural control.
- Do not promote `role_alpha_wasserstein`; it is artifact-compatible but not empirically better here.

## Overclaims To Avoid

- Spectral methods recover lineage or family independently of architecture.
- `paper_spearman` is validated as the best distance.
- Rank/hybrid methods are universally best; the winner is target-dependent.
- kNN association on this balanced slice proves causal spectral-performance signal.
- Canonicalization is only plumbing; schema handling materially changes comparability.
