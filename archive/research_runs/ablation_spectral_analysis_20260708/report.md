# Ablation Spectral Run Analysis

## Sources
- Curation data: `/jumbo/yaoqingyang/kinshuk/Model-Zoo/data/ablation`
- Run artifacts: `/jumbo/yaoqingyang/kinshuk/Model-Zoo/output/ablation`

## Executive Summary
- Curated selected rows: 4713
- Terminal successes: 3125 / 4713 (66.3%)
- Stats artifacts: 3661 / 4713 (77.7%)
- Metrics HDF5 artifacts: 3661 / 4713 (77.7%)
- Selected rows with artifacts but no terminal JSON: 536
- Selected rows with neither artifact nor terminal JSON: 1
- Hierarchy is partly present through `source_model`, `base_model_relation`, `lineage_eval_group`, and `primary_type_bucket`; this report adds a coarse fallback family label from source/model/architecture hints for stratification only.

## Terminal Status
| terminal_status | terminal_stage | terminal_reason | rows |
| --- | --- | --- | --- |
| success | save | completed | 3125 |
| failed | load | model_load_error | 884 |
| not_attempted | n/a | missing_terminal_status | 537 |
| failed | analyze | analysis_exception | 85 |
| failed | supervisor | stale_stage_timeout | 30 |
| failed | analyze | analysis_empty | 27 |
| failed | analyze | cuda_oom | 18 |
| failed | save | save_error | 7 |

## Coverage By Type
| primary_type_bucket | selected_rows | success_rows | failed_rows | success_rate | stats_artifact_rate |
| --- | --- | --- | --- | --- | --- |
| other_text | 3178 | 2353 | 644 | 0.740 | 0.797 |
| finetune | 1174 | 743 | 233 | 0.633 | 0.801 |
| moe | 328 | 29 | 160 | 0.088 | 0.512 |
| merge | 30 | 0 | 12 | 0.000 | 0.600 |
| adapter | 1 | 0 | 1 | 0.000 | 0.000 |
| multimodal | 1 | 0 | 0 | 0.000 | 1.000 |
| quantized | 1 | 0 | 1 | 0.000 | 0.000 |

## Coverage By Lineage Group
| lineage_eval_group | selected_rows | success_rows | failed_rows | success_rate | stats_artifact_rate |
| --- | --- | --- | --- | --- | --- |
| unrelated_negative_pair | 4462 | 3065 | 983 | 0.687 | 0.780 |
| same_base_finetune | 220 | 60 | 55 | 0.273 | 0.745 |
| known_merge_family | 30 | 0 | 12 | 0.000 | 0.600 |
| adapter_versus_merged_base_pair | 1 | 0 | 1 | 0.000 | 0.000 |

## Top Coarse Families
| coarse_family | selected_rows | success_rows | failed_rows | success_rate | stats_artifact_rate |
| --- | --- | --- | --- | --- | --- |
| llama | 1836 | 1212 | 412 | 0.660 | 0.776 |
| mistral | 1492 | 1184 | 232 | 0.794 | 0.845 |
| mixtral | 326 | 29 | 160 | 0.089 | 0.509 |
| qwen | 324 | 138 | 80 | 0.426 | 0.750 |
| gpt2 | 108 | 94 | 14 | 0.870 | 0.870 |
| gemma | 93 | 84 | 9 | 0.903 | 0.903 |
| tinyllama | 91 | 70 | 20 | 0.769 | 0.780 |
| phi | 80 | 45 | 32 | 0.562 | 0.600 |
| unknown | 64 | 45 | 19 | 0.703 | 0.703 |
| gpt-neox | 49 | 45 | 4 | 0.918 | 0.918 |
| opt | 42 | 38 | 4 | 0.905 | 0.905 |
| codellama | 35 | 21 | 14 | 0.600 | 0.600 |
| pythia | 35 | 33 | 2 | 0.943 | 0.943 |
| bloom | 27 | 24 | 3 | 0.889 | 0.889 |
| falcon | 25 | 0 | 25 | 0.000 | 0.000 |
| gptj | 21 | 21 | 0 | 1.000 | 1.000 |
| mpt | 11 | 1 | 10 | 0.091 | 0.091 |
| starcoder | 9 | 3 | 6 | 0.333 | 0.333 |
| xglm | 7 | 6 | 1 | 0.857 | 0.857 |
| gptbigcode | 5 | 5 | 0 | 1.000 | 1.000 |

## Strongest Spectral-Score Correlations
| target | feature | n | spearman | pearson |
| --- | --- | --- | --- | --- |
| MATH Lvl 5 | log_norm_global_mean | 594 | 0.512 | 0.418 |
| Average ⬆️ | spectral_norm_global_mean | 3621 | -0.495 | -0.055 |
| Average ⬆️ | log_norm_global_mean | 3622 | -0.451 | -0.398 |
| MATH Lvl 5 | spectral_norm_global_mean | 593 | 0.446 | 0.318 |
| MMLU-PRO | log_norm_global_mean | 594 | 0.434 | 0.330 |
| Average ⬆️ | num_layers_observed | 3622 | 0.432 | 0.332 |
| Average ⬆️ | alpha_global_std | 3621 | 0.432 | 0.116 |
| MATH Lvl 5 | alpha_mlp_minus_attn | 588 | -0.419 | -0.408 |
| MATH Lvl 5 | alpha_attn_mean | 592 | 0.381 | 0.196 |
| BBH | alpha_mlp_minus_attn | 588 | -0.380 | -0.273 |
| MUSR | num_layers_observed | 594 | 0.375 | 0.356 |
| MATH Lvl 5 | alpha_global_mean | 593 | 0.371 | 0.068 |
| MATH Lvl 5 | alpha_global_mean_abs_distance_to_2 | 593 | 0.370 | 0.068 |
| BBH | num_layers_observed | 594 | 0.362 | 0.529 |
| MATH Lvl 5 | alpha_global_median | 593 | 0.352 | 0.055 |
| MATH Lvl 5 | alpha_global_median_abs_distance_to_2 | 593 | 0.352 | 0.055 |
| GPQA | num_layers_observed | 594 | 0.350 | 0.491 |
| BBH | log_norm_global_mean | 594 | 0.344 | 0.284 |
| Average ⬆️ | valid_alpha_count | 3622 | 0.343 | 0.253 |
| Average ⬆️ | alpha_global_mean_abs_distance_to_2 | 3621 | 0.332 | 0.038 |

## Interpretation Notes
- The paper's most relevant lesson is to treat PL_Alpha_Hill as an ESD shape signature and inspect module families separately before over-collapsing.
- This run is suitable for broad spectral/profile analysis of successful models, but success is not uniform by type. MoE and large/complex models have lower completion rates, so downstream comparisons should report coverage alongside any spectral trend.
- Known lineage is sparse in the selected set; fine-tune and merge conclusions should be treated as higher-confidence where `lineage_eval_group` is known and exploratory where only coarse family hints are available.
