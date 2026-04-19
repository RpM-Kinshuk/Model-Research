# Benchmark Expansion Cycle 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test the smallest honest expansion of the strict root/base benchmark without changing the retrieval methods.

**Architecture:** Keep the current strict freeze and method stack fixed. Run one sensitivity benchmark with `min_family_size=5` to see whether a few additional official root/base families can enter cleanly, then compare that run against the current strict headline baseline and decide whether the relaxed cutoff is worth keeping.

**Tech Stack:** Python 3, `conda run -n esd_ind`, existing `scripts/spectral_similarity_research.py`, CSV artifacts under `research_runs/`

---

### Task 1: Run The Smallest Benchmark Expansion Sensitivity

**Files:**
- Create: `research_runs/spectral_family_retrieval_strict_headline_min5_v1/`
- Modify: none
- Test: command output and generated `report.md`

- [ ] **Step 1: Run the expanded benchmark with the existing strict freeze and a 5-row family cutoff**

Run:

```bash
conda run -n esd_ind python scripts/spectral_similarity_research.py \
  --freeze-csv data/manual_audits/spectral_family_retrieval_label_freeze.csv \
  --min-family-size 5 \
  --output-dir research_runs/spectral_family_retrieval_strict_headline_min5_v1
```

Expected:
- command exits successfully
- writes the full benchmark artifact set into `research_runs/spectral_family_retrieval_strict_headline_min5_v1`

- [ ] **Step 2: Read the expanded run report**

Run:

```bash
sed -n '1,220p' research_runs/spectral_family_retrieval_strict_headline_min5_v1/report.md
```

Expected:
- report shows a larger supported-family count than the current strict headline run if any near-miss families were admitted
- retrieval metrics stay strong enough to compare against the current baseline

### Task 2: Compare Against The Current Strict Headline Baseline

**Files:**
- Create: `research_runs/spectral_family_retrieval_strict_headline_min5_v1/comparison_notes.md`
- Modify: none
- Test: comparison note content and benchmark CSVs

- [ ] **Step 1: Compare split size and retrieval metrics**

Run:

```bash
python3 -c "import pandas as pd; base='research_runs/spectral_family_retrieval_strict_headline_v2/retrieval_metrics.csv'; exp='research_runs/spectral_family_retrieval_strict_headline_min5_v1/retrieval_metrics.csv'; b=pd.read_csv(base); e=pd.read_csv(exp); cols=['method','slice','num_queries','top1_accuracy','mrr','recall@1','recall@3','recall@5']; print('BASE'); print(b[cols].to_string(index=False)); print('EXP'); print(e[cols].to_string(index=False))"
```

Expected:
- clear side-by-side metrics for the same method set
- enough evidence to decide whether the extra families are worth the relaxed cutoff

- [ ] **Step 2: Write a short comparison note**

Write `research_runs/spectral_family_retrieval_strict_headline_min5_v1/comparison_notes.md` with:

```md
# Min-5 Expansion Comparison Notes

- Baseline run: `research_runs/spectral_family_retrieval_strict_headline_v2`
- Expansion run: `research_runs/spectral_family_retrieval_strict_headline_min5_v1`
- Decision question: does relaxing `min_family_size` from `8` to `5` add useful official root/base families without materially weakening retrieval quality?

## Record
- Supported families:
- Supported queries:
- Newly admitted families:
- `full_module_trace` top-1 / MRR delta:
- `attention_trace` top-1 / MRR delta:
- Keep or reject the min-5 expansion:
- Why:
```

Expected:
- one short artifact that records the expansion outcome

### Task 3: Record The Next Bottleneck

**Files:**
- Modify: `docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md`
- Test: note includes the min-5 expansion result and keeps metadata cleanup as the next priority

- [ ] **Step 1: Append a short update to the findings note**

Add a short subsection stating:

```md
## Benchmark Expansion Cycle 1

We tested a minimal benchmark expansion by relaxing the strict root/base family cutoff from `8` to `5` while keeping the same strict freeze and the same retrieval methods.

Outcome:
- [fill in outcome after the run]

Interpretation:
- if the expansion helps and stays clean, it becomes a candidate benchmark variant
- if it does not help enough, the next priority remains metadata cleanup on the confirmed strict slice
```

- [ ] **Step 2: Verify the note reads cleanly**

Run:

```bash
sed -n '1,260p' docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md
```

Expected:
- note includes the new subsection
- no placeholders remain after the run
