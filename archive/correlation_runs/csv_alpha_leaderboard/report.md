# Alpha vs Leaderboard Correlation Report

## Dataset summary
- Matched unique models: 374
- Common-schema matched models: 344
- Duplicate leaderboard rows collapsed before matching: 64
- PCA explained variance: 0.432, 0.127, 0.079

## Strongest raw alpha correlations by target
- MATH Lvl 5: alpha_start_across_modules (Spearman=0.456, q=0.000, n=374)
- #Params (B): alpha_pc1 (Spearman=-0.365, q=0.000, n=374)
- MUSR: alpha_end_module_spread (Spearman=-0.364, q=0.000, n=374)
- downloads: alpha_start_across_modules (Spearman=-0.294, q=0.000, n=367)
- Average ⬆️: alpha_median_module_spread (Spearman=0.270, q=0.000, n=374)
- MMLU-PRO: alpha_median_module_spread (Spearman=0.266, q=0.000, n=374)
- IFEval: alpha_global_fraction_gt_2 (Spearman=0.250, q=0.000, n=374)
- likes: alpha_start_across_modules (Spearman=-0.226, q=0.000, n=367)
- BBH: alpha_end_module_spread (Spearman=-0.210, q=0.000, n=374)
- CO₂ cost (kg): alpha_std_across_modules (Spearman=-0.185, q=0.002, n=374)
- GPQA: alpha_end_module_spread (Spearman=-0.178, q=0.002, n=374)

## Strongest size-adjusted alpha correlations by target
- MATH Lvl 5: attn_k__start (partial Spearman=0.645, q=0.000, covariates=#Params (B), Generation, n=353)
- #Params (B): attn_o__mean (partial Spearman=-0.639, q=0.000, covariates=Generation, n=374)
- MMLU-PRO: mlp_gate__start (partial Spearman=0.457, q=0.000, covariates=#Params (B), Generation, n=371)
- Average ⬆️: attn_v__median (partial Spearman=0.453, q=0.000, covariates=#Params (B), Generation, n=347)
- BBH: attn_k__median (partial Spearman=0.407, q=0.000, covariates=#Params (B), Generation, n=353)
- CO₂ cost (kg): mlp_down__start (partial Spearman=0.372, q=0.000, covariates=#Params (B), Generation, n=371)
- MUSR: attn_k__median (partial Spearman=0.366, q=0.000, covariates=#Params (B), Generation, n=353)
- GPQA: mlp_up__end (partial Spearman=0.304, q=0.000, covariates=#Params (B), Generation, n=367)
- IFEval: attn_v__median (partial Spearman=0.282, q=0.000, covariates=#Params (B), Generation, n=347)
- downloads: attn_k__slope (partial Spearman=0.248, q=0.000, covariates=#Params (B), Generation, n=346)
- likes: attn_k__slope (partial Spearman=0.149, q=0.012, covariates=#Params (B), Generation, n=346)

## Notes
- Raw correlations can reflect model size and generation effects, so the partial-Spearman section adjusts for `#Params (B)` and `Generation` when possible.
- The alpha feature space is built from the same canonical summary representation used in the clustering dashboard: per-module summary stats over sanitized normalized-depth traces plus module-presence features.
- This run also includes alpha-distance-to-2 features, motivated by the HTSR/SETOL interpretation that deviations on either side of 2 can indicate different non-ideal training regimes.
