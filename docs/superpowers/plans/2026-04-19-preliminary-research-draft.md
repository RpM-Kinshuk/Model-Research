# Preliminary Research Draft Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write a markdown-first research draft that captures the current preliminary study as a claims-and-evidence document rather than a submission-ready paper.

**Architecture:** Use the consolidated findings note and the approved research-draft design as the only factual sources. Write one markdown document organized by the approved sections: north star, why we started narrow, what we built, what holds up, what stayed weak, current method hierarchy, canonical artifacts, and remaining work before scale-up. Keep the tone plain, research-facing, and easy to revise after the larger-scale rerun.

**Tech Stack:** Markdown, existing notes, existing experiment artifacts

---

### Task 1: Gather The Canonical Inputs

**Files:**
- Read: `docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md`
- Read: `docs/superpowers/specs/2026-04-19-preliminary-paper-outline.md`
- Read: `docs/superpowers/specs/2026-04-19-preliminary-research-draft-design.md`

- [ ] **Step 1: Extract the claims that survived the preliminary phase**

Capture:
- the main zoo-management framing
- the role of the strict benchmark
- the role of the lean mixed-zoo benchmark
- the strongest retrieval methods
- the current prediction/trend limits

- [ ] **Step 2: Confirm the draft will stay within the approved scope**

The draft must:
- stay markdown-first
- avoid polished submission tone
- avoid new claims not already supported

### Task 2: Write The Markdown Draft

**Files:**
- Create: `docs/superpowers/specs/2026-04-19-preliminary-research-draft.md`
- Source: `docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md`
- Source: `docs/superpowers/specs/2026-04-19-preliminary-research-draft-design.md`

- [ ] **Step 1: Draft the research-facing structure**

Required sections:
- `North Star`
- `Why We Started Narrow`
- `What We Built`
- `What Holds Up`
- `What Stayed Secondary Or Weak`
- `Current Method Hierarchy`
- `Canonical Preliminary Artifact Set`
- `What Needs To Happen Before Scale-Up`

- [ ] **Step 2: Keep the strongest methods centered**

The draft must name:
- `full_module_trace` as the strongest general retrieval method
- `role_aggregated_trace` as the strongest current heterogeneity-aware method

It must also keep:
- prediction secondary
- trend diagnostics descriptive-only
- rejection outside the main claim

### Task 3: Verify And Finalize

**Files:**
- Verify: `docs/superpowers/specs/2026-04-19-preliminary-research-draft.md`
- Verify against: `docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md`
- Verify against: `docs/superpowers/specs/2026-04-19-preliminary-research-draft-design.md`

- [ ] **Step 1: Run a focused coverage check**

Run:
`rg -n "North Star|Why We Started Narrow|What We Built|What Holds Up|What Stayed Secondary Or Weak|Current Method Hierarchy|Canonical Preliminary Artifact Set|What Needs To Happen Before Scale-Up|full_module_trace|role_aggregated_trace|prediction|trend|rejection" docs/superpowers/specs/2026-04-19-preliminary-research-draft.md`

Expected:
- all required sections appear
- primary method names appear
- prediction, trend, and rejection are present but not centered

- [ ] **Step 2: Review for overclaiming**

Confirm the draft does not say:
- rejection is solved
- open-world identification is solved
- general performance prediction is solved
- the whole zoo-management problem is solved

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-04-19-preliminary-research-draft.md docs/superpowers/plans/2026-04-19-preliminary-research-draft.md
git commit -m "docs: add preliminary research draft"
```
