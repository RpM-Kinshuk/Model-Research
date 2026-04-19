# Preliminary Paper Outline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write a paper-facing outline note for the preliminary study that stays aligned with the consolidated findings and the original zoo-management-first research goals.

**Architecture:** Use the existing consolidated findings note as the factual source of truth, then distill it into a separate paper-outline document with short section summaries. Keep the outline narrow: introduction, setup, methods, benchmarks, results, limitations, and next step. Do not add new experiments or new claims.

**Tech Stack:** Markdown, local research notes, existing experiment artifacts

---

### Task 1: Map Source Material

**Files:**
- Read: `docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md`
- Read: `reference-work/sections/abstract.tex`
- Read: `reference-work/sections/methods.tex`
- Read: `reference-work/sections/results/model lineage.tex`

- [ ] **Step 1: Extract the sections that matter for the paper-facing story**

Capture:
- main claim
- supporting claim
- non-claims
- benchmark split logic
- strongest method results
- remaining preliminary limits

- [ ] **Step 2: Confirm the paper outline will stay aligned with the current narrative**

Check that the outline centers:
- zoo management first
- heterogeneous support as a real requirement
- performance prediction as secondary

### Task 2: Write The Outline Note

**Files:**
- Create: `docs/superpowers/specs/2026-04-19-preliminary-paper-outline.md`
- Source: `docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md`

- [ ] **Step 1: Draft the outline with short section summaries**

Required sections:
- Title / framing
- Introduction
- Problem Setup
- Methods
- Benchmarks
- Results
- What Holds Up
- Limits
- Next Step

- [ ] **Step 2: Keep the claims disciplined**

Make sure the note:
- names `full_module_trace` and `role_aggregated_trace` as the primary methods
- keeps prediction secondary
- keeps trend diagnostics descriptive-only
- does not present rejection as a solved result

### Task 3: Verify and Finalize

**Files:**
- Verify: `docs/superpowers/specs/2026-04-19-preliminary-paper-outline.md`
- Verify against: `docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md`

- [ ] **Step 1: Run a consistency check**

Verify the outline does not contradict:
- benchmark descriptions
- headline metrics
- method hierarchy
- stated limitations

- [ ] **Step 2: Run focused verification**

Run:
`rg -n "full_module_trace|role_aggregated_trace|prediction|trend|rejection|heterogeneous" docs/superpowers/specs/2026-04-19-preliminary-paper-outline.md`

Expected:
- all major topics appear
- the primary method names appear
- prediction and trend are present but clearly secondary

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-04-19-preliminary-paper-outline.md docs/superpowers/plans/2026-04-19-preliminary-paper-outline.md
git commit -m "docs: add preliminary paper outline"
```
