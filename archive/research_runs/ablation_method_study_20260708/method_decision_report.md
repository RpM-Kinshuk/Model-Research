# Ablation Method Study

## Decision
- Best family-retrieval AUROC: `structural_basics`.
- Best kNN prediction method by average per-target MAE rank: `hybrid_rank`.
- Best top-1 family recall: `structural_basics`.

## Pragmatic Verdict
- Family retrieval is dominated by structural metadata; treat family AUROC as a leakage-prone control, not a spectral discovery.
- Existing alpha baselines are stronger by family AUROC; report paper Spearman as the required paper-faithful baseline, not the operational default.
- Paper Spearman has better top-1 family recall than existing alpha baselines, which supports keeping it in retrieval studies.
- Rank-shape methods are the strongest direction for kNN benchmark prediction in this study; they keep the paper's rank-invariance idea while adding smoother similarity.
- Role-wise alpha Wasserstein is a useful sanity check, but it does not beat the leading methods here.

## Family Retrieval
| method                 | pairwise_auc | top1_family_recall | eligible_queries |
| ---------------------- | ------------ | ------------------ | ---------------- |
| structural_basics      | 0.8495       | 0.9623             | 398.0000         |
| existing_scalar_l2     | 0.7477       | 0.8266             | 398.0000         |
| role_alpha_wasserstein | 0.7410       | 0.8141             | 398.0000         |
| rank_rbf               | 0.7259       | 0.8518             | 398.0000         |
| hybrid_rank            | 0.7240       | 0.8518             | 398.0000         |
| paper_spearman         | 0.7191       | 0.8618             | 398.0000         |
| existing_role_l2       | 0.6986       | 0.7613             | 398.0000         |

## kNN Prediction
| target     | method                 | n   | mae     | spearman |
| ---------- | ---------------------- | --- | ------- | -------- |
| Average ⬆️ | hybrid_rank            | 391 | 5.3075  | 0.7962   |
| Average ⬆️ | paper_spearman         | 391 | 5.3782  | 0.7908   |
| Average ⬆️ | rank_rbf               | 391 | 5.4211  | 0.8036   |
| Average ⬆️ | existing_scalar_l2     | 391 | 5.9738  | 0.7625   |
| Average ⬆️ | role_alpha_wasserstein | 391 | 5.9843  | 0.7621   |
| Average ⬆️ | structural_basics      | 391 | 6.4078  | 0.7871   |
| Average ⬆️ | existing_role_l2       | 391 | 6.6282  | 0.6954   |
| MATH Lvl 5 | hybrid_rank            | 43  | 6.1179  | 0.8139   |
| MATH Lvl 5 | rank_rbf               | 43  | 6.3390  | 0.8021   |
| MATH Lvl 5 | existing_role_l2       | 43  | 6.3428  | 0.7888   |
| MATH Lvl 5 | paper_spearman         | 43  | 6.9052  | 0.7267   |
| MATH Lvl 5 | role_alpha_wasserstein | 43  | 7.2934  | 0.7345   |
| MATH Lvl 5 | existing_scalar_l2     | 43  | 9.4064  | 0.6238   |
| MATH Lvl 5 | structural_basics      | 43  | 10.9657 | 0.4630   |
| MMLU-PRO   | rank_rbf               | 43  | 6.9602  | 0.7308   |
| MMLU-PRO   | paper_spearman         | 43  | 7.0601  | 0.6825   |
| MMLU-PRO   | existing_role_l2       | 43  | 7.2153  | 0.6706   |
| MMLU-PRO   | hybrid_rank            | 43  | 7.4692  | 0.7106   |
| MMLU-PRO   | role_alpha_wasserstein | 43  | 7.9688  | 0.6658   |
| MMLU-PRO   | existing_scalar_l2     | 43  | 10.1985 | 0.4089   |
| MMLU-PRO   | structural_basics      | 43  | 10.1989 | 0.3697   |

## Newer Methods Considered
- Raw-weight CKA/fingerprint methods such as AWM require access to weight matrices, so they are future-work here rather than valid baselines for saved alpha summaries: https://arxiv.org/abs/2510.06738
- Attention-product SVD fingerprints such as GhostSpec require internal attention weight products, not the current HDF5 alpha traces: https://arxiv.org/abs/2511.06390
- Gradient fingerprints such as TensorGuard require loading models and computing gradient responses, which is outside this artifact-only study: https://arxiv.org/abs/2506.01631
- Learned model-hub representations are promising for a larger supervised system, but they need a separate training corpus of model weights/features: https://arxiv.org/abs/2510.02096

## Caveats
- `coarse_family` is a pragmatic proxy label inferred from metadata/name/architecture, not a manually audited ground truth.
- Methods requiring raw weights or gradients were not tested here because the available reusable artifacts are alpha/stat summaries.
