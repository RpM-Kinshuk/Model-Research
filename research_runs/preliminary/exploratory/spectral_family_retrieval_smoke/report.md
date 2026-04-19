# Spectral Family Retrieval Preliminary Report

Headline claim tested: module-aware spectral candidate generation for base-family retrieval with abstention.

## Included methods
- metadata_only
- pl_alpha_scalar
- attention_trace
- full_module_trace
- summary_module

## Split summary
- seed: 0
- min_family_size: 8
- gallery_supported: 0
- queries_supported: 0
- queries_unsupported: 687
- queries_heterogeneous: 97

## Retrieval metrics
| method | slice | num_queries | top1_accuracy | mrr | recall@1 | recall@3 | recall@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| metadata_only | all | 0 | n/a | n/a | n/a | n/a | n/a |
| metadata_only | architecture_matched | 0 | n/a | n/a | n/a | n/a | n/a |
| metadata_only | size_matched | 0 | n/a | n/a | n/a | n/a | n/a |
| pl_alpha_scalar | all | 0 | n/a | n/a | n/a | n/a | n/a |
| pl_alpha_scalar | architecture_matched | 0 | n/a | n/a | n/a | n/a | n/a |
| pl_alpha_scalar | size_matched | 0 | n/a | n/a | n/a | n/a | n/a |
| attention_trace | all | 0 | n/a | n/a | n/a | n/a | n/a |
| attention_trace | architecture_matched | 0 | n/a | n/a | n/a | n/a | n/a |
| attention_trace | size_matched | 0 | n/a | n/a | n/a | n/a | n/a |
| full_module_trace | all | 0 | n/a | n/a | n/a | n/a | n/a |
| full_module_trace | architecture_matched | 0 | n/a | n/a | n/a | n/a | n/a |
| full_module_trace | size_matched | 0 | n/a | n/a | n/a | n/a | n/a |
| summary_module | all | 0 | n/a | n/a | n/a | n/a | n/a |
| summary_module | architecture_matched | 0 | n/a | n/a | n/a | n/a | n/a |
| summary_module | size_matched | 0 | n/a | n/a | n/a | n/a | n/a |

## Rejection metrics
| method | slice | num_examples | threshold | accepted_precision | accepted_coverage | auroc | auprc | fpr95 | aurc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metadata_only | all | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| metadata_only | architecture_matched | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| metadata_only | size_matched | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| pl_alpha_scalar | all | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| pl_alpha_scalar | architecture_matched | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| pl_alpha_scalar | size_matched | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| attention_trace | all | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| attention_trace | architecture_matched | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| attention_trace | size_matched | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| full_module_trace | all | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| full_module_trace | architecture_matched | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| full_module_trace | size_matched | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| summary_module | all | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| summary_module | architecture_matched | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| summary_module | size_matched | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Family neighborhood top-k
_No rows._

## Heterogeneous stress bucket
- rows: 97
