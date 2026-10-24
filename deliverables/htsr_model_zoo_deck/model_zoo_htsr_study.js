"use strict";

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { imageSizingContain } = require("./pptxgenjs_helpers/image");
const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./pptxgenjs_helpers/layout");

const HERE = __dirname;
const ASSET = (name) => path.join(HERE, "assets", name);
const DATA = JSON.parse(fs.readFileSync(ASSET("chart_data.json"), "utf8"));

const C = {
  berkeley: "002676",
  booking: "003B95",
  bookingLight: "EAF2FF",
  dartmouth: "00693E",
  dartmouthLight: "E4F3EA",
  gold: "FDB515",
  goldLight: "FFF4C7",
  cyan: "00AEEF",
  ink: "14213D",
  body: "344054",
  muted: "667085",
  line: "D9E2EC",
  surface: "F7F9FC",
  white: "FFFFFF",
  risk: "B42318",
  riskLight: "FEE4E2",
};

const FONT = "Aptos";
const DISPLAY = "Aptos Display";
const W = 13.333;
const H = 7.5;

const pptx = new pptxgen();
pptx.defineLayout({ name: "MODEL_WIDE", width: W, height: H });
pptx.layout = "MODEL_WIDE";
pptx.author = "Model Research";
pptx.company = "Model Research";
pptx.subject = "Exploratory audit of weight-derived grouping, base retrieval, and performance controls";
pptx.title = "Model weights as evidence";
pptx.lang = "en-US";
pptx.theme = { headFontFace: DISPLAY, bodyFontFace: FONT, lang: "en-US" };

const counts = DATA.counts;
const coverage = DATA.profile_sources;
const groupingCoverage = coverage.grouping_coverage?.find((row) => row.collection === "Combined");
if (!groupingCoverage) {
  throw new Error("Grouping coverage is missing; regenerate chart_data.json with make_charts.py");
}
const collections = DATA.collections;
const internal = DATA.internal;
const REVIEW = DATA.reassessment;
const REVIEW_FIELDS = {
  performance: ["population", "generation", "split", "method", "n_models", "mae"],
  tasks: ["population", "generation", "split", "method", "target", "mae"],
  lineage_gallery: ["cohort", "method", "candidate_count", "n_present", "n_absent_operator", "n_supplied_unprofiled", "micro_top1"],
  open_set: ["cohort", "method", "contrast", "signal", "auroc"],
  all_matrix_clusters: ["label_source", "adjusted_rand"],
};
if (!REVIEW || typeof REVIEW.run_id !== "string" || !REVIEW.run_id.trim()
    || !Number.isFinite(REVIEW.counts?.all_matrix_alpha_models)) {
  throw new Error("chart_data.json requires DATA.reassessment.run_id and all-matrix coverage counts");
}
for (const [table, fields] of Object.entries(REVIEW_FIELDS)) {
  if (!Array.isArray(REVIEW[table]) || !REVIEW[table].length
      || REVIEW[table].some((row) => !row || fields.some((field) => row[field] === undefined))) {
    throw new Error(`DATA.reassessment.${table} requires rows with: ${fields.join(", ")}`);
  }
}
const manual = DATA.validation.find((row) => row.label_source === "manual_confirmed");
const consensus = DATA.validation.find(
  (row) => row.label_source === "metadata_root_consensus"
);
const graphRoot = DATA.validation.find(
  (row) => row.label_source === "metadata_graph_root"
);

function n(value) {
  return Number(value).toLocaleString("en-US");
}

function f(value, digits = 3) {
  return Number(value).toFixed(digits);
}

function pct(value, digits = 1) {
  return `${(100 * Number(value)).toFixed(digits)}%`;
}

function reviewRow(table, fields) {
  const rows = REVIEW[table].filter((row) =>
    Object.entries(fields).every(([key, value]) => row[key] === value)
  );
  if (rows.length !== 1) {
    throw new Error(`Expected one ${table} row: ${JSON.stringify(fields)}`);
  }
  return rows[0];
}

function reviewPerf(generation, method, split = "near_group_0995") {
  return reviewRow("performance", {
    population: "matched_role_population", generation, split, method,
  });
}

function reviewGallery(method) {
  return reviewRow("lineage_gallery", { cohort: "common_queries_and_gallery", method });
}

function reviewAuc(method, signal) {
  return reviewRow("open_set", {
    cohort: "common_queries_and_gallery", method, signal,
    contrast: "profiled_root_vs_all_unavailable",
  }).auroc;
}

const rankGallery = reviewGallery("parent_rank32_dks");
const traceGallery = reviewGallery("trace_sqrt_norm_pearson");

function addText(slide, text, x, y, w, h, options = {}) {
  slide.addText(text, {
    x,
    y,
    w,
    h,
    fontFace: options.fontFace || FONT,
    fontSize: options.fontSize || 13,
    color: options.color || C.body,
    margin: options.margin === undefined ? 0 : options.margin,
    valign: options.valign || "top",
    breakLine: false,
    ...options,
  });
}

function addBox(slide, text, x, y, w, h, options = {}) {
  slide.addText(text, {
    x,
    y,
    w,
    h,
    shape: pptx.ShapeType.roundRect,
    rectRadius: 0.05,
    fill: { color: options.fill || C.surface },
    line: { color: options.line || C.line, width: options.lineWidth || 0.8 },
    fontFace: options.fontFace || FONT,
    fontSize: options.fontSize || 12,
    color: options.color || C.body,
    bold: options.bold || false,
    align: options.align || "left",
    valign: options.valign || "mid",
    margin: options.margin === undefined ? 0.14 : options.margin,
  });
}

function addRule(slide, x, y, w, color = C.line, width = 1) {
  slide.addShape(pptx.ShapeType.line, {
    x,
    y,
    w,
    h: 0,
    line: { color, width },
  });
}

function addArrow(slide, x, y, w, color = C.muted) {
  slide.addShape(pptx.ShapeType.line, {
    x,
    y,
    w,
    h: 0,
    line: { color, width: 1.5, endArrowType: "triangle" },
  });
}

function addImage(slide, name, x, y, w, h) {
  slide.addImage({
    path: ASSET(name),
    ...imageSizingContain(ASSET(name), x, y, w, h),
  });
}

function addDot(slide, x, y, diameter, color, transparency = 0) {
  slide.addShape(pptx.ShapeType.ellipse, {
    x: x - diameter / 2,
    y: y - diameter / 2,
    w: diameter,
    h: diameter,
    fill: { color, transparency },
    line: { transparency: 100 },
  });
}

function addCoverArtwork(slide) {
  const palette = [C.cyan, "4E79B7", C.gold, C.cyan, "6D8EC4"];
  for (let row = 0; row < 7; row += 1) {
    for (let column = 0; column < 9; column += 1) {
      const keep = (row * 9 + column * 5) % 11 !== 4;
      if (!keep) continue;
      const x = 8.72 + column * 0.34 + 0.025 * Math.sin(row * 2.1 + column);
      const y = 1.03 + row * 0.32 + 0.018 * Math.cos(column * 1.8 - row);
      const diameter = 0.055 + 0.018 * ((row * 3 + column * 2) % 4);
      addDot(slide, x, y, diameter, palette[(row + column * 2) % palette.length]);
    }
  }

  slide.addShape(pptx.ShapeType.line, {
    x: 8.72,
    y: 3.62,
    w: 2.68,
    h: 0,
    line: { color: C.gold, width: 1.7, endArrowType: "triangle" },
  });

  const galaxies = [
    { x: 9.12, y: 4.72, color: C.dartmouth, phase: 0.2, count: 17 },
    { x: 10.48, y: 4.49, color: C.gold, phase: 1.5, count: 19 },
    { x: 11.61, y: 4.96, color: C.cyan, phase: 2.7, count: 18 },
  ];
  galaxies.forEach((galaxy, galaxyIndex) => {
    addDot(slide, galaxy.x, galaxy.y, 0.145, galaxy.color);
    for (let index = 0; index < galaxy.count; index += 1) {
      const arm = index % 3;
      const step = Math.floor(index / 3) + 1;
      const radius = 0.13 + step * (0.09 + galaxyIndex * 0.006);
      const angle = galaxy.phase + arm * ((2 * Math.PI) / 3) + step * 0.48;
      const x = galaxy.x + radius * Math.cos(angle);
      const y = galaxy.y + 0.48 * radius * Math.sin(angle);
      const diameter = 0.025 + 0.011 * ((index + galaxyIndex) % 3);
      const color = index % 7 === 0 ? C.gold : galaxyIndex === 0 ? C.dartmouth : C.cyan;
      addDot(slide, x, y, diameter, color, index % 4 === 0 ? 18 : 0);
    }
  });
}

function addStat(slide, value, label, x, y, w, color = C.booking, note = "", onDark = false) {
  addText(slide, value, x, y, w, 0.48, {
    fontFace: DISPLAY,
    fontSize: 27,
    bold: true,
    color,
  });
  addText(slide, label, x, y + 0.53, w, 0.32, {
    fontSize: 10.5,
    bold: true,
    color: onDark ? C.white : C.ink,
  });
  if (note) {
    addText(slide, note, x, y + 0.9, w, 0.4, {
      fontSize: 8.4,
      color: onDark ? "C9D6E8" : C.muted,
    });
  }
}

function addBullet(slide, text, x, y, w, h, color = C.body, fontSize = 11) {
  slide.addText(text, {
    x,
    y,
    w,
    h,
    fontFace: FONT,
    fontSize,
    color,
    margin: 0,
    valign: "top",
    bullet: { type: "bullet", indent: fontSize, hanging: fontSize * 0.35 },
  });
}

function baseSlide(kicker, title, accent, page, source) {
  const slide = pptx.addSlide();
  slide.background = { color: C.white };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: 6.25,
    h: 0.055,
    fill: { color: C.berkeley },
    line: { transparency: 100 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 6.25,
    y: 0,
    w: 3.65,
    h: 0.055,
    fill: { color: C.dartmouth },
    line: { transparency: 100 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 9.9,
    y: 0,
    w: 3.433,
    h: 0.055,
    fill: { color: C.gold },
    line: { transparency: 100 },
  });
  addText(slide, kicker, 0.58, 0.22, 5.4, 0.22, {
    fontSize: 9,
    bold: true,
    color: accent,
    charSpacing: 0,
  });
  addText(slide, title, 0.58, 0.49, 12.15, 0.46, {
    fontFace: DISPLAY,
    fontSize: 25,
    bold: true,
    color: C.ink,
    valign: "mid",
  });
  addRule(slide, 0.58, 1.06, 12.15, C.line, 0.8);
  addRule(slide, 0.58, 7.17, 12.15, C.line, 0.7);
  addText(slide, source, 0.58, 7.24, 10.9, 0.13, {
    fontSize: 6.6,
    color: C.muted,
    valign: "mid",
  });
  addText(slide, String(page).padStart(2, "0"), 12.08, 7.22, 0.64, 0.15, {
    fontSize: 8,
    bold: true,
    color: accent,
    align: "right",
    valign: "mid",
  });
  return slide;
}

// 1. Title
{
  const slide = pptx.addSlide();
  slide.background = { color: C.berkeley };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: 8.3,
    h: 0.075,
    fill: { color: C.gold },
    line: { transparency: 100 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 8.3,
    y: 0,
    w: 2.7,
    h: 0.075,
    fill: { color: C.dartmouth },
    line: { transparency: 100 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 11,
    y: 0,
    w: 2.333,
    h: 0.075,
    fill: { color: C.cyan },
    line: { transparency: 100 },
  });
  addText(slide, "MODEL-ZOO / HEAVY-TAILED SELF-REGULARIZATION", 0.72, 0.62, 7.2, 0.25, {
    fontSize: 10,
    bold: true,
    color: C.gold,
    charSpacing: 0,
  });
  addText(slide, "Model weights\nas evidence", 0.72, 1.18, 7.2, 1.52, {
    fontFace: DISPLAY,
    fontSize: 40,
    bold: true,
    color: C.white,
  });
  addText(
    slide,
    "Exploratory audit: weight grouping, base retrieval, and performance controls",
    0.75,
    3.03,
    7.55,
    0.48,
    { fontSize: 17, color: "D9E2EC" }
  );
  addStat(slide, n(counts.lineage_models), "lineage models", 0.76, 4.45, 2.2, C.gold, "Phase 2 + ablation", true);
  addStat(slide, n(counts.unlabelled_clusters), "weight-derived groups", 3.36, 4.45, 2.15, C.cyan, "selected without labels", true);
  addStat(
    slide,
    f(reviewPerf("modern", "nonalpha_spectral_trees").mae, 2),
    "non-alpha modern MAE",
    5.92,
    4.45,
    2.0,
    "7ACBA5",
    "near-group held-out audit",
    true
  );
  addCoverArtwork(slide);
  addText(slide, "September 2026 results / October 2026 scale-up plan", 0.75, 6.48, 7.4, 0.24, {
    fontSize: 10,
    color: "D9E2EC",
  });
  addText(slide, `Exploratory reassessment: ${REVIEW.run_id}`, 0.75, 6.93, 7.4, 0.18, {
    fontSize: 7.8,
    color: "B9C5D6",
  });
}

// 2. Boundary
{
  const slide = baseSlide(
    "STUDY BOUNDARY",
    "Keep predictor inputs separate from evaluation labels",
    C.berkeley,
    2,
    `Sources: ${REVIEW.run_id}/protocol.json; parent prediction freezes`
  );
  const columns = [
    {
      x: 0.62,
      color: C.booking,
      fill: C.bookingLight,
      title: "Unlabeled discovery",
      input: "complete core + calibrated partial roles",
      output: "clusters + representative models",
    },
    {
      x: 4.48,
      color: C.dartmouth,
      fill: C.dartmouthLight,
      title: "Known base candidates",
      input: "profiles + fixed candidate list",
      output: "ranked candidate bases",
    },
    {
      x: 8.34,
      color: "9A6A00",
      fill: C.goldLight,
      title: "Performance",
      input: "matrix summaries + training scores",
      output: "held-out score estimates",
    },
  ];
  columns.forEach((item) => {
    addText(slide, item.title, item.x, 1.36, 3.4, 0.36, {
      fontSize: 16,
      bold: true,
      color: item.color,
    });
    addBox(slide, `INPUT\n${item.input}`, item.x, 1.95, 3.4, 0.9, {
      fill: item.fill,
      line: item.color,
      color: C.ink,
      bold: true,
      fontSize: 12,
      align: "center",
    });
    addArrow(slide, item.x + 1.08, 3.11, 1.24, item.color);
    addBox(slide, `FROZEN OUTPUT\n${item.output}`, item.x, 3.43, 3.4, 0.9, {
      fill: C.white,
      line: item.color,
      color: C.ink,
      bold: true,
      fontSize: 12,
      align: "center",
    });
  });
  addBox(
    slide,
    "EVALUATION ONLY  |  declared source roots and family labels score frozen lineage predictions; held-out benchmark scores score predictions",
    0.62,
    5.13,
    11.72,
    0.72,
    {
      fill: C.riskLight,
      line: C.risk,
      color: C.risk,
      bold: true,
      align: "center",
      fontSize: 11,
    }
  );
  addText(
    slide,
    "Collection metadata shaped selection; collection IDs define one holdout. IDs join records and break ties, not predictor features. This is exploratory reuse, not untouched confirmation.",
    0.66,
    6.15,
    11.65,
    0.54,
    { fontSize: 10.3, color: C.muted, align: "center" }
  );
}

// 3. Study map
{
  const slide = baseSlide(
    "STUDY MAP",
    "Four phases, each ending in a checked output",
    C.berkeley,
    3,
    "Sources: collection code, parent weight-only run, and exploratory reassessment protocol."
  );
  const phases = [
    {
      x: 0.62,
      number: "1",
      title: "Assemble model lists",
      detail: "Phase 2 and ablation IDs\ncanonicalize and remove duplicates",
      output: `${n(collections.phase2.selected + collections.ablation.selected)} selected records`,
      color: C.booking,
      fill: C.bookingLight,
    },
    {
      x: 3.73,
      number: "2",
      title: "Analyze weights",
      detail: "read checkpoint matrices\ncompute alpha by layer and role",
      output: `${n(counts.lineage_models)} lineage / ${n(counts.performance_profiled_models)} performance`,
      color: C.berkeley,
      fill: C.surface,
    },
    {
      x: 6.84,
      number: "3",
      title: "Make predictions",
      detail: "form groups · rank known bases\nestimate benchmark scores",
      output: "predictions saved before labels",
      color: C.dartmouth,
      fill: C.dartmouthLight,
    },
    {
      x: 9.95,
      number: "4",
      title: "Validate",
      detail: "score against declared source links\nmeasure held-out benchmark error",
      output: "reported results and limits",
      color: "9A6A00",
      fill: C.goldLight,
    },
  ];
  phases.forEach((phase, index) => {
    addBox(slide, phase.number, phase.x, 1.38, 0.42, 0.42, {
      fill: phase.color,
      line: phase.color,
      color: C.white,
      fontSize: 13,
      bold: true,
      align: "center",
      margin: 0,
    });
    addText(slide, phase.title, phase.x + 0.58, 1.38, 2.1, 0.42, {
      fontSize: 14,
      bold: true,
      color: phase.color,
      valign: "mid",
    });
    addBox(slide, phase.detail, phase.x, 2.13, 2.72, 1.15, {
      fill: phase.fill,
      line: phase.color,
      color: C.ink,
      fontSize: 10.7,
      bold: true,
      align: "center",
    });
    addArrow(slide, phase.x + 0.72, 3.59, 1.25, phase.color);
    addBox(slide, phase.output, phase.x, 3.9, 2.72, 0.78, {
      fill: C.white,
      line: phase.color,
      color: C.ink,
      fontSize: 10.4,
      bold: true,
      align: "center",
    });
    if (index < phases.length - 1) addArrow(slide, phase.x + 2.79, 2.69, 0.25, C.muted);
  });
  addBox(
    slide,
    "Predictor features come from saved weights; score fitting uses training outcomes. Collection metadata shaped curation and one holdout, so the full process is not metadata-free.",
    0.72,
    5.55,
    11.9,
    0.72,
    {
      fill: C.berkeley,
      line: C.berkeley,
      color: C.white,
      fontSize: 11,
      bold: true,
      align: "center",
    }
  );
}

// 4. Dataset curation
{
  const slide = baseSlide(
    "DATASET CURATION",
    "Two collections were built for different jobs",
    C.booking,
    4,
    "Sources: fetch_data/build_phase2_zoo.py, fetch_data/build_ablation_dataset.py, and ablation_source_manifest.json"
  );
  addImage(slide, "corpus_sources.png", 0.42, 1.24, 7.25, 3.08);

  addText(slide, "PHASE 2 · BROAD MODEL ZOO", 7.95, 1.31, 4.45, 0.25, {
    fontSize: 10,
    bold: true,
    color: C.booking,
  });
  const phase2Steps = [
    "Combine the Model Atlas graph, sampled records, leaderboard rows, and targeted Hub searches.",
    "Normalize repository IDs and declared revision fields; the analyzed checkpoint revision is not verified.",
    "Fill model-type quotas while balancing score and lineage availability; favor rows with existing weight results.",
  ];
  phase2Steps.forEach((item, index) =>
    addBox(slide, `${index + 1}  ${item}`, 7.95, 1.72 + index * 0.73, 4.62, 0.56, {
      fill: index === 0 ? C.bookingLight : C.white,
      line: C.booking,
      color: C.ink,
      fontSize: 9.2,
      bold: true,
      align: "left",
      margin: 0.1,
    })
  );

  addText(slide, "ABLATION · DISJOINT SCORE TEST", 7.95, 4.18, 4.45, 0.25, {
    fontSize: 10,
    bold: true,
    color: C.dartmouth,
  });
  addBox(
    slide,
    `${n(collections.ablation.candidates)} leaderboard candidates`,
    7.95,
    4.58,
    1.78,
    0.58,
    { fill: C.dartmouthLight, line: C.dartmouth, color: C.ink, bold: true, align: "center", fontSize: 10.3 }
  );
  addArrow(slide, 9.87, 4.87, 0.42, C.dartmouth);
  addBox(
    slide,
    `${n(collections.ablation.selected)} selected`,
    10.43,
    4.58,
    2.12,
    0.58,
    { fill: C.dartmouth, line: C.dartmouth, color: C.white, bold: true, align: "center", fontSize: 11.2 }
  );
  addBox(
    slide,
    `Removed: ${n(collections.ablation.phase2_overlap_rows)} Phase 2 overlaps · ${n(collections.ablation.inaccessible_rows)} inaccessible repositories · ${n(collections.ablation.duplicate_rows)} lower-priority duplicates`,
    7.95,
    5.38,
    4.6,
    0.68,
    { fill: C.riskLight, line: C.risk, color: C.risk, bold: true, align: "center", fontSize: 9.2 }
  );
  addText(
    slide,
    "Keep score coverage high, rotate across model types, cap missing scores at 15%, and stop below 5,000 when no more rows pass.",
    7.95,
    6.23,
    4.6,
    0.54,
    { fontSize: 9.5, color: C.body, bold: true, align: "center" }
  );
  addBox(
    slide,
    "Collection metadata shapes coverage. Predictor features exclude these labels, but source selection and joins remain part of the evidence boundary.",
    0.62,
    5.14,
    6.82,
    0.78,
    {
      fill: C.goldLight,
      line: C.gold,
      color: C.ink,
      fontSize: 10.1,
      bold: true,
      align: "center",
    }
  );
}

// 5. Dataset distribution
{
  const slide = baseSlide(
    "DATASET DISTRIBUTION",
    "Selection buckets describe curation; ablation is benchmark-heavy",
    C.dartmouth,
    5,
    "Sources: completed run's Phase 2 and ablation source tables; architecture graph used descriptively after selection."
  );
  addImage(slide, "corpus_types.png", 0.38, 1.17, 6.15, 3.82);
  addImage(slide, "corpus_architectures.png", 6.72, 1.17, 6.15, 3.82);
  addBox(
    slide,
    `PHASE 2 · ${n(collections.phase2.selected)}\n${n(collections.phase2.benchmark_rows)} average-score records · ${n(collections.phase2.lineage_metadata_rows)} declared source links\n${n(collections.phase2.architecture_rows)} architecture labels · ${n(collections.phase2.parameter_rows)} parameter counts`,
    0.62,
    5.22,
    5.82,
    0.88,
    {
      fill: C.bookingLight,
      line: C.booking,
      color: C.ink,
      fontSize: 10.1,
      bold: true,
      align: "center",
    }
  );
  addBox(
    slide,
    `ABLATION · ${n(collections.ablation.selected)}\n${n(collections.ablation.benchmark_rows)} average-score records · ${n(collections.ablation.lineage_metadata_rows)} declared source links\n${n(collections.ablation.architecture_rows)} architecture labels · ${n(collections.ablation.parameter_rows)} parameter counts`,
    6.88,
    5.22,
    5.82,
    0.88,
    {
      fill: C.dartmouthLight,
      line: C.dartmouth,
      color: C.ink,
      fontSize: 10.1,
      bold: true,
      align: "center",
    }
  );
  addText(
    slide,
    "Type-chart categories are selection buckets, not verified model types. Source links are declared, not verified ancestry; architecture labels are descriptive.",
    0.72,
    6.38,
    11.9,
    0.3,
    { fontSize: 9.5, color: C.muted, align: "center" }
  );
}

// 6. Data flow
{
  const slide = baseSlide(
    "DATA FLOW",
    "Shared weight checks lead to task-specific coverage",
    C.booking,
    6,
    `Sources: ${DATA.run_id}/population_ledger.csv; source-run provenance.json and exclusions.csv`
  );
  const rows = [
    {
      y: 1.42,
      name: "PHASE 2",
      listed: coverage.phase2_listed,
      analyzed: coverage.phase2_analyzed,
      ready: coverage.phase2_profiles,
      strict: coverage.phase2_strict,
      color: C.booking,
      fill: C.bookingLight,
    },
    {
      y: 3.66,
      name: "ABLATION",
      listed: coverage.ablation_listed,
      analyzed: coverage.ablation_analyzed,
      ready: coverage.ablation_profiles,
      strict: coverage.ablation_strict,
      color: C.dartmouth,
      fill: C.dartmouthLight,
    },
  ];
  rows.forEach((row) => {
    addBox(slide, `${row.name}\n${n(row.listed)} listed`, 0.66, row.y, 2.25, 0.94, {
      fill: row.fill,
      line: row.color,
      color: C.ink,
      fontSize: 12,
      bold: true,
      align: "center",
    });
    addArrow(slide, 3.03, row.y + 0.47, 0.55, row.color);
    addBox(slide, `${n(row.analyzed)}\nweight results found`, 3.72, row.y, 2.35, 0.94, {
      fill: C.surface,
      line: C.muted,
      color: C.ink,
      fontSize: 11.5,
      bold: true,
      align: "center",
    });
    addArrow(slide, 6.19, row.y + 0.47, 0.55, row.color);
    addBox(slide, `${n(row.ready)}\nlineage models`, 6.88, row.y, 2.35, 0.94, {
      fill: row.fill,
      line: row.color,
      color: C.ink,
      fontSize: 11.5,
      bold: true,
      align: "center",
    });
    addArrow(slide, 9.35, row.y + 0.47, 0.48, row.color);
  });
  addBox(slide, `${n(counts.performance_profiled_models)}\nperformance profiles`, 9.98, 2.24, 2.55, 1.62, {
    fill: C.goldLight,
    line: C.gold,
    color: C.ink,
    fontSize: 17,
    bold: true,
    align: "center",
  });
  addText(
    slide,
    `Strict seven-role core: ${n(counts.strict_seven_role_models)} models. Reliable missing-role patterns add ${n(counts.lineage_extension_models)} lineage assignments. Performance uses the label-free cutoff of at least ${n(counts.minimum_usable_roles)} observed roles.`,
    0.72,
    5.36,
    11.85,
    0.5,
    { fontSize: 10.5, color: C.body, bold: true, align: "center" }
  );
  addBox(
    slide,
    "These are parent role-profile counts. The reassessment also uses all-matrix summaries; source IDs define the collection holdout, not predictor features.",
    0.72,
    6.05,
    11.85,
    0.58,
    {
      fill: C.surface,
      line: C.line,
      color: C.ink,
      fontSize: 10.2,
      bold: true,
      align: "center",
    }
  );
}

// 7. Profile
{
  const slide = baseSlide(
    "WEIGHT ANALYSIS",
    "Compare models using the layer roles that are actually observed",
    C.booking,
    7,
    "Sources: stored DKS alpha values by layer; archive/reference-work/Spectral Signatures of Large Language Models.html"
  );
  const steps = [
    ["W", "weight matrix"],
    ["W^T W", "eigenvalues"],
    ["alpha", "DKS tail fit"],
    ["1-7 roles", "ordered layers"],
    ["role ranks", "observed-role summary"],
  ];
  steps.forEach(([label, note], index) => {
    const x = 0.65 + index * 2.47;
    addBox(slide, label, x, 1.48, 1.75, 0.68, {
      fill: index === 4 ? C.booking : C.surface,
      line: index === 4 ? C.booking : C.line,
      color: index === 4 ? C.white : C.ink,
      bold: true,
      fontSize: 16,
      align: "center",
    });
    addText(slide, note, x, 2.3, 1.75, 0.3, {
      fontSize: 9.4,
      color: C.muted,
      align: "center",
    });
    if (index < steps.length - 1) addArrow(slide, x + 1.85, 1.82, 0.47);
  });
  addBox(
    slide,
    "Attention: query · key · value · output",
    0.68,
    3.05,
    5.78,
    0.57,
    { fill: C.bookingLight, line: C.booking, color: C.ink, bold: true, align: "center" }
  );
  addBox(
    slide,
    "Feed-forward: gate · up · down",
    6.88,
    3.05,
    5.78,
    0.57,
    { fill: C.dartmouthLight, line: C.dartmouth, color: C.ink, bold: true, align: "center" }
  );
  addText(slide, "THIS STUDY", 0.68, 4.08, 1.6, 0.25, {
    fontSize: 10,
    bold: true,
    color: C.booking,
  });
  addText(
    slide,
    "Custom stored DKS alpha, not a WeightWatcher run · depth scaled to [0,1] · 32 positions per observed role · within-role ranks for grouping",
    0.68,
    4.48,
    5.78,
    1.03,
    { fontSize: 11, color: C.body }
  );
  addText(slide, "REFERENCE PAPER", 6.88, 4.08, 2.3, 0.25, {
    fontSize: 10,
    bold: true,
    color: C.dartmouth,
  });
  addText(
    slide,
    "Fixed-tail Hill alpha, k=n/2 · raw-alpha clustering · role-wise Spearman similarity · order-preserving DP for unequal depths · MoE experts averaged",
    6.88,
    4.48,
    5.78,
    1.03,
    { fontSize: 11, color: C.body }
  );
  addBox(
    slide,
    "Ranks discard alpha offsets, scales, and spacing, but retain order. Absolute alpha and other saved matrix statistics can still carry performance signal.",
    0.68,
    5.92,
    11.98,
    0.62,
    {
      fill: C.goldLight,
      line: C.gold,
      color: C.ink,
      bold: true,
      align: "center",
      fontSize: 10.8,
    }
  );
}

// 8. Corpus
{
  const slide = baseSlide(
    "DATA",
    "How much of the collection can layer-wise grouping cover?",
    C.booking,
    8,
    `Sources: ${DATA.run_id}/population_ledger.csv; retained combined-study provenance.json`
  );
  addImage(slide, "corpus_coverage.png", 0.42, 1.38, 8.02, 5.38);
  addStat(slide, n(groupingCoverage.grouped), "grouped models", 8.85, 1.42, 3.0, C.booking);
  addStat(slide, pct(groupingCoverage.share_analyzed), "of analyzed models", 8.85, 2.68, 3.0, C.dartmouth,
    `${n(groupingCoverage.grouped)} / ${n(groupingCoverage.analyzed)}`);
  addStat(slide, pct(groupingCoverage.share_selected), "of selected records", 8.85, 3.99, 3.0, C.berkeley,
    `${n(groupingCoverage.grouped)} / ${n(groupingCoverage.selected)}`);
  addBox(
    slide,
    `Fixed seven-role vocabulary; only tested missing-role patterns are admitted. ${n(groupingCoverage.no_analysis)} selected records have no completed analysis. Unseen-architecture accuracy remains untested.`,
    8.82,
    5.46,
    3.55,
    1.15,
    {
      fill: C.surface,
      line: C.line,
      color: C.body,
      fontSize: 10.5,
      bold: true,
      align: "center",
    }
  );
}

// 9. Discovery
{
  const slide = baseSlide(
    "LINEAGE DISCOVERY",
    `${n(counts.unlabelled_clusters)} core groups give the clearest separation without family labels`,
    C.dartmouth,
    9,
    `Sources: ${DATA.run_id}/lineage_cluster_selection.csv and lineage_internal_metrics.csv`
  );
  addImage(slide, "cluster_selection.png", 0.46, 1.24, 7.95, 5.48);
  addStat(slide, f(internal.silhouette_cosine, 3), "group separation", 8.72, 1.5, 1.7, C.booking, "higher is better");
  addStat(slide, f(internal.mean_seed_stability_ari, 3), "repeat-run agreement", 10.67, 1.5, 1.82, C.dartmouth, "average of 3 reruns");
  addStat(slide, f(internal.minimum_seed_stability_ari, 3), "lowest rerun agreement", 8.72, 3.08, 1.7, "9A6A00");
  addStat(slide, f(internal.davies_bouldin_euclidean, 3), "within/between spread", 10.67, 3.08, 1.82, C.muted, "lower is better");
  addBox(
    slide,
    `HOW ${n(counts.unlabelled_clusters)} WAS CHOSEN\n\nFit candidate counts on\n${n(internal.n_core_models)} complete profiles\n\nKeep the count with\nthe best separation`,
    8.72,
    4.7,
    3.77,
    1.45,
    {
      fill: C.dartmouthLight,
      line: C.dartmouth,
      color: C.ink,
      fontSize: 11.2,
      bold: true,
      align: "center",
    }
  );
  addText(slide, `${n(internal.n_extension_models)} partial profiles are attached only after label-free masking checks pass.`, 8.75, 6.32, 3.7, 0.4, {
    fontSize: 10,
    bold: true,
    color: C.dartmouth,
    align: "center",
  });
}

// 10. Validation
{
  const slide = baseSlide(
    "VALIDATION AFTER PREDICTION",
    "Validate grouping against overlapping, imperfect label views",
    C.dartmouth,
    10,
    `Sources: parent lineage_validation_metrics.csv; ${REVIEW.run_id}/all_matrix_cluster_validation.csv`
  );
  addImage(slide, fs.existsSync(ASSET("review_clustering.png")) ? "review_clustering.png" : "validation_agreement.png", 0.42, 1.26, 7.28, 4.82);
  addText(slide, "LABELS OPENED AFTER GROUPING", 7.96, 1.37, 4.5, 0.24, {
    fontSize: 10,
    bold: true,
    color: C.dartmouth,
  });
  const labelSets = [
    ["Human-reviewed", `${n(manual.n_models)} models assigned to ${n(manual.n_families)} broad base families by manual review.`, C.bookingLight, C.booking],
    ["Metadata agreement", `${n(consensus.n_models)} models assigned to ${n(consensus.n_families)} broad families when metadata rules agree.`, C.dartmouthLight, C.dartmouth],
    ["Declared roots", `${n(graphRoot.n_models)} models linked to ${n(graphRoot.n_families)} graph-component roots; not verified historical ancestry.`, C.goldLight, "9A6A00"],
  ];
  labelSets.forEach(([title, detail, fill, line], index) => {
    const y = 1.82 + index * 1.19;
    addBox(slide, title, 7.96, y, 1.45, 0.72, {
      fill,
      line,
      color: C.ink,
      fontSize: 10.5,
      bold: true,
      align: "center",
    });
    addText(slide, detail, 9.62, y + 0.02, 2.78, 0.67, {
      fontSize: 9.4,
      color: C.body,
      valign: "mid",
    });
  });
  const metrics = [
    ["ARI", "Pairwise grouping agreement, corrected for chance. 1 means identical grouping.", C.bookingLight, C.booking],
    ["Shared information", "How much knowing one grouping tells us about the other. 1 is complete.", C.dartmouthLight, C.dartmouth],
    ["Purity", "Majority-label share inside each predicted group. It can favor too many small groups.", C.goldLight, "9A6A00"],
  ];
  metrics.forEach(([title, detail, fill, line], index) => {
    const x = 0.62 + index * 4.05;
    addBox(slide, `${title}\n${detail}`, x, 6.08, 3.7, 0.72, {
      fill,
      line,
      color: C.ink,
      fontSize: 8.6,
      bold: true,
      align: "center",
      margin: 0.08,
    });
  });
}

// 11. Known candidate bases
{
  const slide = baseSlide(
    "KNOWN-CANDIDATE LINEAGE TEST",
    "Base matching depends on the comparison and treatment of ties",
    C.berkeley,
    11,
    `Sources: ${REVIEW.run_id}/lineage_gallery_metrics.csv; audit/gallery_tie_sensitivity.csv. Trace: arxiv.org/html/2608.07786v1`
  );
  addImage(slide, "review_gallery.png", 0.42, 1.26, 7.55, 5.12);
  addStat(slide, `${n(traceGallery.candidate_count)} / ${n(DATA.gallery.n_supplied_roots)}`, "available / supplied bases", 8.35, 1.42, 1.95, C.berkeley);
  addStat(slide, n(traceGallery.n_present), "matched known-root queries", 10.62, 1.42, 1.98, C.booking);
  addStat(slide, pct(rankGallery.micro_top1, 2), "rank top-1", 8.35, 2.8, 1.95, C.booking);
  addStat(slide, pct(traceGallery.micro_top1, 2), "Trace top-1", 10.62, 2.8, 1.98, C.dartmouth);
  addBox(
    slide,
    "Trace compares weight magnitude through layers, averaging seven role-wise correlations. Its first-choice gain is sensitive to tied candidates; the full and tie-excluded subsets answer different questions.",
    8.35,
    4.03,
    4.02,
    1.2,
    {
      fill: C.bookingLight,
      line: C.booking,
      color: C.ink,
      fontSize: 10,
      bold: true,
      align: "center",
    }
  );
  addBox(
    slide,
    `${n(traceGallery.n_absent_operator)} other labeled queries have a root absent from the supplied list; ${n(traceGallery.n_supplied_unprofiled)} have a listed root without usable measurements. These are queries, not additional candidate bases.`,
    8.35,
    5.48,
    4.02,
    1.0,
    {
      fill: C.riskLight,
      line: C.risk,
      color: C.risk,
      fontSize: 9.8,
      bold: true,
      align: "center",
    }
  );
  addText(slide, `After excluding exact ties in either method: rank ${pct(reviewRow("gallery_ties", {cohort: "neither_method_has_exact_tie", method: "parent_rank32_dks"}).micro_top1, 2)}, Trace ${pct(reviewRow("gallery_ties", {cohort: "neither_method_has_exact_tie", method: "trace_sqrt_norm_pearson"}).micro_top1, 2)}. This changes the model/root mix; equal-weight-per-root accuracy still favors Trace.`, 0.66, 6.48, 7.24, 0.42, {
    fontSize: 9.2, color: C.risk, bold: true,
  });
}

// 12. Paper comparison
{
  const slide = baseSlide(
    "PAPER COMPARISON",
    "Our larger study is not an exact replication",
    C.gold,
    12,
    `Sources: arxiv.org/html/2607.03377v1, Sec. 4.2 and 5.3; ${REVIEW.run_id}/protocol.json`
  );
  const rows = [
    ["Dimension", "Spectral Signatures paper", "Exploratory reassessment"],
    ["Prediction boundary", "Weight similarities; training scores for prediction", "Weight features; training-only score fitting"],
    ["Alpha estimator", "Fixed-tail Hill, k = n/2", "Custom stored DKS; not WeightWatcher or fixed Hill"],
    ["Grouping / similarity", "Flattened raw alpha / role-wise Spearman", "Rank grouping retained; separate Trace base test"],
    ["Unequal depth", "Z-score + order-preserving matching", "Ranks: 32-point interpolation; Trace: pair-max depth"],
    ["Performance setup", "Five folds stated; 100 half-splits in results", "Five fixed near-group parts; explicit stress splits"],
    ["Scope of comparison", "Different candidates, cohorts and targets", "No matched replication; no claim to beat the paper"],
  ];
  const x = [0.62, 3.3, 7.78];
  const w = [2.68, 4.48, 4.92];
  let y = 1.33;
  rows.forEach((values, rowIndex) => {
    const height = rowIndex === 0 ? 0.46 : 0.66;
    values.forEach((value, columnIndex) => {
      addText(slide, value, x[columnIndex], y, w[columnIndex], height, {
        fontSize: rowIndex === 0 ? 10 : 10.2,
        bold: rowIndex === 0 || columnIndex === 0,
        color: rowIndex === 0 ? C.white : columnIndex === 0 ? C.ink : C.body,
        fill: { color: rowIndex === 0 ? C.berkeley : columnIndex === 0 ? C.surface : C.white },
        line: { color: C.line, width: 0.7 },
        margin: 0.1,
        valign: "mid",
      });
    });
    y += height;
  });
  addBox(
    slide,
    "The paper's split-description discrepancy is unresolved. Neither its accuracy nor our MAE establishes superiority across different populations and protocols.",
    0.68,
    6.35,
    11.98,
    0.58,
    {
      fill: C.goldLight,
      line: C.gold,
      color: C.ink,
      bold: true,
      align: "center",
      fontSize: 10.5,
    }
  );
}

// 13. Performance method
{
  const slide = baseSlide(
    "PERFORMANCE METHOD",
    "Keep similar models together in five test parts",
    C.booking,
    13,
    `Sources: ${REVIEW.run_id}/protocol.json, splits.csv and performance_predictions.csv`
  );
  addText(slide, "BUILD FIVE TEST PARTS", 0.72, 1.34, 4.2, 0.26, {
    fontSize: 10,
    bold: true,
    color: C.booking,
  });
  const setupSteps = [
    ["Connect near profiles\nsimilarity >= 0.995", C.bookingLight, C.booking],
    ["Keep each group intact\nin five fixed parts", C.surface, C.muted],
    ["Freeze the same splits\nfor predictor contrasts", C.dartmouthLight, C.dartmouth],
  ];
  setupSteps.forEach(([label, fill, line], index) => {
    const x = 0.76 + index * 4.13;
    addBox(slide, label, x, 1.78, 3.15, 0.82, {
      fill,
      line,
      color: C.ink,
      bold: true,
      align: "center",
      fontSize: 11.6,
    });
    if (index < 2) addArrow(slide, x + 3.3, 2.19, 0.52, line);
  });
  addText(slide, "REPEAT FIVE TIMES", 0.72, 3.25, 4.2, 0.26, {
    fontSize: 10,
    bold: true,
    color: C.dartmouth,
  });
  const testSteps = [
    ["Hold out\n1 part", C.goldLight, "9A6A00"],
    ["Use the other\n4 parts", C.dartmouthLight, C.dartmouth],
    ["Fit preprocessing\non training rows only", C.bookingLight, C.booking],
    ["Fit fixed trees / ridge\nor rank 3-neighbor model", C.surface, C.muted],
    ["Open the held-out\nscore and measure error", C.riskLight, C.risk],
  ];
  testSteps.forEach(([label, fill, line], index) => {
    const x = 0.72 + index * 2.5;
    addBox(slide, label, x, 3.7, 2.02, 0.86, {
      fill,
      line,
      color: C.ink,
      bold: true,
      align: "center",
      fontSize: 10.6,
    });
    if (index < 4) addArrow(slide, x + 2.12, 4.13, 0.27, line);
  });
  addBox(slide, "Fixed trees: 200 iterations, 15 leaves, rate 0.05, L2=10, minimum leaf=20. Ridge penalty=100; no held-out tuning.", 0.76, 5.25, 5.75, 0.84, {
    fill: C.bookingLight,
    line: C.booking,
    color: C.ink,
    fontSize: 10.3,
    bold: true,
    align: "center",
  });
  addBox(slide, "Controls: shape, non-alpha spectra, combined spectra, and rank neighbors. Average-score targets and individual tasks are evaluated separately.", 6.8, 5.25, 5.75, 0.84, {
    fill: C.dartmouthLight,
    line: C.dartmouth,
    color: C.ink,
    fontSize: 10.3,
    bold: true,
    align: "center",
  });
  addText(
    slide,
    "Training scores are permitted; held-out scores are for evaluation. Collection labels define one stress split. Previously studied data: exploratory, not untouched confirmation.",
    0.72,
    6.39,
    11.9,
    0.43,
    { fontSize: 10.2, color: C.muted, align: "center" }
  );
}

// 14. Matched performance controls and individual tasks
{
  const slide = baseSlide(
    "PERFORMANCE CONTROLS",
    "Alpha does not improve the key performance controls",
    C.booking,
    14,
    `Sources: ${REVIEW.run_id}/performance_metrics.csv and performance_task_metrics.csv; near_group_0995, matched_role_population`
  );
  addImage(slide, "review_performance.png", 0.42, 1.22, 7.44, 4.76);
  addImage(slide, "review_tasks.png", 8.03, 1.26, 4.74, 3.71);
  const pair = (method) => ["modern", "legacy"].map((generation) => f(reviewPerf(generation, method).mae)).join(" / ");
  addBox(slide, `AVERAGE-SCORE MAE: MODERN / LEGACY\nNon-alpha trees ${pair("nonalpha_spectral_trees")}\nCombined trees ${pair("role_spectral_trees")}`, 8.12, 5.18, 4.43, 1.03, {
    fill: C.dartmouthLight, line: C.dartmouth, color: C.ink,
    fontSize: 10.5, bold: true, align: "center",
  });
  addText(slide, `Shape ${pair("matrix_shape_trees")} · rank ${pair("rank_knn3")}\n${n(reviewPerf("modern", "rank_knn3").n_models)} modern / ${n(reviewPerf("legacy", "rank_knn3").n_models)} legacy rows; same near-group splits.`, 0.68, 5.96, 7.0, 0.5, {
    fontSize: 10, color: C.muted, align: "center",
  });
  addBox(
    slide,
    "Alpha-specific gain is not established. Combined = alpha plus other matrix measurements; individual-task checks do not replace the controls.",
    0.68,
    6.57,
    11.92,
    0.42,
    {
      fill: C.goldLight,
      line: C.gold,
      color: C.ink,
      bold: true,
      align: "center",
      fontSize: 10,
      margin: 0.06,
    }
  );
}

// 15. Stress test
{
  const slide = baseSlide(
    "NEW-REGION STRESS TEST",
    "Unseen groups and collections expose larger errors",
    C.risk,
    15,
    `Sources: ${REVIEW.run_id}/performance_metrics.csv; audit prediction_metrics_audit.csv and collection_direction_metrics.csv`
  );
  addImage(slide, "review_stress.png", 0.44, 1.2, 7.64, 5.72);
  const modern = reviewPerf("modern", "nonalpha_spectral_ridge", "whole_weight_cluster");
  const legacy = reviewPerf("legacy", "nonalpha_spectral_ridge", "whole_weight_cluster");
  addText(slide, "NON-ALPHA RIDGE · WHOLE-GROUP MAE", 8.35, 1.42, 4.07, 0.44, {
    fontSize: 10.8, bold: true, color: C.booking,
  });
  addStat(slide, f(modern.mae), "modern", 8.35, 2.04, 1.83, C.booking);
  addStat(slide, f(legacy.mae), "legacy", 10.57, 2.04, 1.83, C.dartmouth);
  addBox(
    slide,
    "Whole weight clusters are held out together. This is a shift test, not the near-group interpolation test; compare each method with the training median.",
    8.35,
    3.42,
    4.05,
    0.97,
    {
      fill: C.bookingLight,
      line: C.booking,
      color: C.ink,
      fontSize: 10.5,
      bold: true,
      align: "center",
    }
  );
  addBox(
    slide,
    `COLLECTION-HOLDOUT WARNING\nCombined ridge produces ${n(reviewRow("range_checks", {population: "matched_role_population", generation: "modern", split: "collection_holdout", method: "role_spectral_ridge"}).n_prediction_outside_0_100)} modern predictions outside 0-100. These are range failures, not plausible benchmark forecasts.`,
    8.35,
    4.76,
    4.05,
    1.08,
    {
      fill: C.riskLight,
      line: C.risk,
      color: C.risk,
      fontSize: 10.4,
      bold: true,
      align: "center",
    }
  );
  addText(
    slide,
    "Errors change across held-out groups and collections. Inspect the score range and coverage, not only average error.",
    8.35,
    6.19,
    4.05,
    0.56,
    { fontSize: 10.2, color: C.risk, bold: true, align: "center" }
  );
}

// 16. Decision
{
  const slide = baseSlide(
    "CONCLUSION",
    "Keep rank grouping, test Trace bases, and retain non-alpha controls",
    C.berkeley,
    16,
    `Sources: ${REVIEW.run_id}/protocol.json, audit tables, and frozen lineage evaluation`
  );
  const columns = [
    {
      x: 0.62,
      title: "Retain",
      color: C.dartmouth,
      fill: C.dartmouthLight,
      items: [
        "Keep layer-wise grouping: all-matrix summaries cover more models but separate families poorly",
        "Trace as a base-retrieval comparison, with fixed gallery and tie diagnostics",
        `Non-alpha trees as baseline: MAE ${f(reviewPerf("modern", "nonalpha_spectral_trees").mae)} modern / ${f(reviewPerf("legacy", "nonalpha_spectral_trees").mae)} legacy`,
      ],
    },
    {
      x: 4.48,
      title: "Interpretation",
      color: C.booking,
      fill: C.bookingLight,
      items: [
        "Ranks discard magnitude; direct measurements predict scores better",
        "Adding alpha did not improve the key spectral controls",
        "Local prediction and broad grouping are distinct from historical ancestry",
      ],
    },
    {
      x: 8.34,
      title: "Remaining uncertainty",
      color: "9A6A00",
      fill: C.goldLight,
      items: [
        "Repository-ID source joins do not verify the analyzed checkpoint revision",
        "Missing bases and near-tied candidates limit exact-ID retrieval",
        "Whole-group shifts worsen errors; collection-shift ridge can leave 0-100",
      ],
    },
  ];
  columns.forEach((column) => {
    addText(slide, column.title, column.x, 1.36, 3.4, 0.38, {
      fontSize: 17,
      bold: true,
      color: column.color,
    });
    addRule(slide, column.x, 1.91, 3.4, column.color, 2.2);
    column.items.forEach((item, index) => {
      addBox(slide, item, column.x, 2.28 + index * 1.02, 3.4, 0.76, {
        fill: column.fill,
        line: column.color,
        color: C.ink,
        fontSize: 10.7,
        bold: true,
        align: "center",
      });
    });
  });
  addBox(
    slide,
    "Next experiment: use the newer pinned collection to test whether weight measurements improve task-specific model choices. See the October scale-up plan in the appendix.",
    0.62,
    6.24,
    11.72,
    0.68,
    {
      fill: C.berkeley,
      line: C.berkeley,
      color: C.white,
      fontSize: 11.2,
      bold: true,
      align: "center",
    }
  );
}

// 17. Appendix divider
{
  const slide = pptx.addSlide();
  slide.background = { color: C.berkeley };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: 5.9,
    h: 0.07,
    fill: { color: C.gold },
    line: { transparency: 100 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 5.9,
    y: 0,
    w: 3.1,
    h: 0.07,
    fill: { color: C.dartmouth },
    line: { transparency: 100 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 9.0,
    y: 0,
    w: 4.333,
    h: 0.07,
    fill: { color: C.cyan },
    line: { transparency: 100 },
  });
  addText(slide, "APPENDIX", 0.72, 0.72, 2.1, 0.3, {
    fontSize: 10,
    bold: true,
    color: C.gold,
    charSpacing: 0,
  });
  addText(slide, "Methods and next study", 0.72, 1.27, 8.7, 0.78, {
    fontFace: DISPLAY,
    fontSize: 38,
    bold: true,
    color: C.white,
    valign: "mid",
  });
  addText(
    slide,
    "Operational details, next weight-based methods, and terms used in the main results",
    0.72,
    2.22,
    9.9,
    0.52,
    { fontSize: 16, color: "D6E3F5" }
  );
  const items = [
    ["A1", "Collection and GPU scheduling", C.cyan],
    ["A2", "Matrix analysis and storage", C.gold],
    ["A3", "Future direct weight comparison", "65C18C"],
    ["A4", "Terms and evidence guide", C.white],
    ["A5", "October scale-up research plan", C.gold],
  ];
  items.forEach(([id, label, color], index) => {
    const y = 3.35 + index * 0.72;
    addText(slide, id, 0.78, y, 0.55, 0.3, {
      fontSize: 10.5,
      bold: true,
      color,
      valign: "mid",
    });
    addRule(slide, 1.48, y + 0.15, 0.66, color, 1.8);
    addText(slide, label, 2.34, y - 0.01, 4.2, 0.34, {
      fontSize: 14,
      bold: true,
      color: C.white,
      valign: "mid",
    });
  });
  addText(slide, "17", 12.05, 7.12, 0.62, 0.18, {
    fontSize: 8,
    bold: true,
    color: C.gold,
    align: "right",
  });
}

// 18. Collection and GPU scheduling
{
  const slide = baseSlide(
    "APPENDIX A1 · COLLECTION",
    "One scheduler assigns jobs to available graphics processors",
    C.booking,
    18,
    "Historical collection workflow; newer operational collector: /jumbo/yaoqingyang/kinshuk/Model-Zoo/docs/operations/analysis.md"
  );

  addBox(slide, "Curated model list", 0.58, 1.34, 2.1, 0.7, {
    fill: C.bookingLight,
    line: C.booking,
    color: C.ink,
    fontSize: 12,
    bold: true,
    align: "center",
  });
  addArrow(slide, 2.83, 1.69, 0.55, C.booking);
  addBox(slide, "Scheduler\nchecks GPU memory and reservations", 3.52, 1.25, 2.72, 0.88, {
    fill: C.berkeley,
    line: C.berkeley,
    color: C.white,
    fontSize: 11.2,
    bold: true,
    align: "center",
  });

  const workers = [
    ["Job 1", "model A", 0.65, C.booking],
    ["Job 2", "model B", 2.41, C.dartmouth],
    ["Job 3", "model C", 4.17, "9A6A00"],
    ["Job 4", "model D", 5.93, C.risk],
  ];
  addArrow(slide, 4.88, 2.24, 0.0, C.muted);
  workers.forEach(([job, model, x, color]) => {
    slide.addShape(pptx.ShapeType.line, {
      x: 4.88,
      y: 2.18,
      w: x + 0.67 - 4.88,
      h: 0.67,
      line: { color: C.line, width: 1.2, endArrowType: "triangle" },
    });
    addBox(slide, `${job}\n${model}`, x, 2.92, 1.34, 0.78, {
      fill: C.white,
      line: color,
      color: C.ink,
      fontSize: 10.5,
      bold: true,
      align: "center",
    });
    addArrow(slide, x + 0.38, 3.9, 0.58, color);
    addBox(slide, "assigned\nGPU(s)", x, 4.16, 1.34, 0.7, {
      fill: C.surface,
      line: color,
      color: C.body,
      fontSize: 9.8,
      bold: true,
      align: "center",
    });
  });

  addText(slide, "CONFIGURABLE CONTROLS", 8.08, 1.32, 3.8, 0.25, {
    fontSize: 10,
    bold: true,
    color: C.booking,
  });
  const controls = [
    "which GPU IDs may be used",
    "GPUs per job and jobs at once",
    "memory threshold before assignment",
    "retry count and stale-job timeout",
    "reload, drain, or stop while running",
  ];
  controls.forEach((item, index) =>
    addBox(slide, item, 8.08, 1.72 + index * 0.62, 4.58, 0.46, {
      fill: index % 2 === 0 ? C.bookingLight : C.white,
      line: C.line,
      color: C.ink,
      fontSize: 10.1,
      bold: true,
      align: "left",
      margin: 0.12,
    })
  );
  addBox(
    slide,
    "Each job uses its own download cache. Removing it after the job prevents storage growth, at the cost of downloading shared files again.",
    8.08,
    5.02,
    4.58,
    0.92,
    {
      fill: C.goldLight,
      line: C.gold,
      color: C.ink,
      fontSize: 10.2,
      bold: true,
      align: "center",
    }
  );
  addText(
    slide,
    "The memory check and internal reservation avoid overlap among these jobs; they are not a cluster-wide lock against unrelated users.",
    0.72,
    6.24,
    11.78,
    0.36,
    { fontSize: 9.8, color: C.muted, align: "center" }
  );
}

// 19. Matrix analysis and storage
{
  const slide = baseSlide(
    "APPENDIX A2 · MODEL ANALYSIS",
    "From weight matrices to saved spectral measurements",
    C.dartmouth,
    19,
    "Sources: historical model statistics; current collector: /jumbo/yaoqingyang/kinshuk/Model-Zoo/docs/operations/analysis.md"
  );
  const pipeline = [
    ["W", "weight matrix", C.booking],
    ["s", "singular values", C.berkeley],
    ["s²", "eigenvalues of WᵀW", C.dartmouth],
    ["alpha", "power-law tail fit", "9A6A00"],
    ["record", "metrics + layer role", C.risk],
  ];
  pipeline.forEach(([label, note, color], index) => {
    const x = 0.54 + index * 2.56;
    addBox(slide, label, x, 1.42, 1.75, 0.67, {
      fill: index === 4 ? color : C.white,
      line: color,
      color: index === 4 ? C.white : C.ink,
      fontSize: 16,
      bold: true,
      align: "center",
    });
    addText(slide, note, x - 0.08, 2.24, 1.91, 0.42, {
      fontSize: 9.2,
      color: C.muted,
      align: "center",
    });
    if (index < pipeline.length - 1) addArrow(slide, x + 1.86, 1.76, 0.52, C.muted);
  });

  addText(slide, "PLUG-IN ANALYSIS CHOICES", 0.68, 3.03, 3.3, 0.25, {
    fontSize: 10,
    bold: true,
    color: C.dartmouth,
  });
  addBox(slide, "DKS\nchoose the tail start that best fits", 0.68, 3.47, 2.7, 0.8, {
    fill: C.dartmouthLight,
    line: C.dartmouth,
    color: C.ink,
    fontSize: 10.8,
    bold: true,
    align: "center",
  });
  addBox(slide, "Fixed rules\nuse a specified part of the tail", 3.62, 3.47, 2.7, 0.8, {
    fill: C.bookingLight,
    line: C.booking,
    color: C.ink,
    fontSize: 10.8,
    bold: true,
    align: "center",
  });

  addText(slide, "STRUCTURED OUTPUTS", 7.02, 3.03, 3.3, 0.25, {
    fontSize: 10,
    bold: true,
    color: C.booking,
  });
  const outputs = [
    ["CSV / HDF5", "per-layer metrics and model fields"],
    ["Full spectra", "current collector: saved by default; opt-out available"],
    ["Logs and status", "progress, failures, retries, and timing"],
  ];
  outputs.forEach(([title, note], index) => {
    addBox(slide, `${title}\n${note}`, 7.02 + index * 1.86, 3.47, 1.64, 1.02, {
      fill: index === 1 ? C.goldLight : C.surface,
      line: index === 1 ? C.gold : C.line,
      color: C.ink,
      fontSize: 9.3,
      bold: true,
      align: "center",
      margin: 0.09,
    });
  });
  addBox(
    slide,
    "Matrix work can be split across several GPUs. This lowers wall-clock time when matrix operations dominate, but does not guarantee fewer total GPU-seconds.",
    0.68,
    5.38,
    5.64,
    0.9,
    {
      fill: C.bookingLight,
      line: C.booking,
      color: C.ink,
      fontSize: 10.3,
      bold: true,
      align: "center",
    }
  );
  addBox(
    slide,
    "New collector: pinned revisions and matrix coverage. Historical results here still use stored DKS alpha; the new default uses the fixed-tail rule.",
    7.02,
    5.38,
    5.64,
    0.9,
    {
      fill: C.riskLight,
      line: C.risk,
      color: C.ink,
      fontSize: 10.3,
      bold: true,
      align: "center",
    }
  );
}

// 20. Next method
{
  const slide = baseSlide(
    "APPENDIX A3 · NEXT EVIDENCE",
    "Add direct weight-matrix comparisons for within-family verification",
    C.dartmouth,
    20,
    "Primary sources: https://arxiv.org/html/2608.07786v1; https://arxiv.org/html/2608.14929v1; https://arxiv.org/html/2510.06738v3"
  );
  addBox(
    slide,
    `STAGE 1\nrank grouping + Trace retrieval\n${n(counts.lineage_models)} rank assignments`,
    0.72,
    1.42,
    3.1,
    1.3,
    {
      fill: C.bookingLight,
      line: C.booking,
      color: C.ink,
      fontSize: 13,
      bold: true,
      align: "center",
    }
  );
  addArrow(slide, 3.98, 2.08, 1.0, C.muted);
  addBox(
    slide,
    "STAGE 2\ndirect weight-matrix check\nsmall candidate set",
    5.15,
    1.42,
    3.1,
    1.3,
    {
      fill: C.dartmouthLight,
      line: C.dartmouth,
      color: C.ink,
      fontSize: 13,
      bold: true,
      align: "center",
    }
  );
  addArrow(slide, 8.42, 2.08, 1.0, C.muted);
  addBox(
    slide,
    "OUTPUT\nrelatedness evidence\ncalibrate on separate pairs",
    9.58,
    1.42,
    3.1,
    1.3,
    {
      fill: C.goldLight,
      line: C.gold,
      color: C.ink,
      fontSize: 13,
      bold: true,
      align: "center",
    }
  );
  const papers = [
    {
      x: 0.72,
      id: "arXiv:2608.07786",
      title: "Spectral energy + subspace alignment",
      note: "Trace tested from sqrt(norm). Fine directional comparison requires top singular vectors, not scalar summaries.",
      color: C.booking,
    },
    {
      x: 4.48,
      id: "arXiv:2608.14929",
      title: "Centered residual signatures",
      note: "Center Wout Win by subtracting trace(M) I/d. Requires compatible blocks; symmetric similarity is not ancestry direction.",
      color: C.dartmouth,
    },
    {
      x: 8.24,
      id: "AWM · arXiv:2510.06738",
      title: "Assignment + centered alignment",
      note: "Embedding assignment and sign alignment, then Q/K unbiased CKA. Requires direct weights; not tested in this audit.",
      color: "9A6A00",
    },
  ];
  papers.forEach((paper) => {
    addText(slide, paper.id, paper.x, 3.35, 3.38, 0.25, {
      fontSize: 9.5,
      bold: true,
      color: paper.color,
    });
    addBox(slide, paper.title, paper.x, 3.76, 3.38, 0.72, {
      fill: C.surface,
      line: paper.color,
      color: C.ink,
      bold: true,
      align: "center",
      fontSize: 11,
    });
    addText(slide, paper.note, paper.x, 4.69, 3.38, 0.9, {
      fontSize: 9.6,
      color: C.body,
      align: "center",
    });
  });
  addBox(
    slide,
    "Trace is feasible from saved scalar norms and was tested. Directional, residual, and AWM checks still require unavailable direct-weight information; none alone proves historical ancestry.",
    0.72,
    5.86,
    11.9,
    0.7,
    {
      fill: C.riskLight,
      line: C.risk,
      color: C.risk,
      fontSize: 10.8,
      bold: true,
      align: "center",
    }
  );
}

// 21. Terms and evidence guide
{
  const slide = baseSlide(
    "APPENDIX A4 · DEFINITIONS",
    "Measures used in this study",
    C.booking,
    21,
    "Definitions follow the analysis code and cited reference methods."
  );
  const terms = [
    ["Alpha", "Exponent fitted to the upper eigenvalue distribution of a weight matrix."],
    ["DKS", "Chooses the tail start that minimizes the Kolmogorov-Smirnov fit distance."],
    ["ARI", "Adjusted Rand Index: grouping agreement corrected for chance; 1 is exact."],
    ["Spearman", "Agreement in rank order; 1 means the two rankings match."],
    ["MAE", "Mean absolute error: the average size of a score prediction error."],
    ["Near-group test", "Keep near-profile groups intact in five fixed parts; fit scores using the other four."],
    ["Trace", "Mean role-wise Pearson of Frobenius depth profiles; resample to pair-max depth."],
    ["Representative", "The observed core model closest to its group mean; not necessarily its historical base."],
  ];
  terms.forEach(([term, definition], index) => {
    const column = index < 4 ? 0 : 1;
    const row = index % 4;
    const x = column === 0 ? 0.68 : 6.82;
    const y = 1.35 + row * 1.18;
    addText(slide, term, x, y, 1.45, 0.3, {
      fontSize: 12,
      bold: true,
      color: column === 0 ? C.booking : C.dartmouth,
    });
    addText(slide, definition, x + 1.55, y - 0.01, 4.22, 0.68, {
      fontSize: 10.3,
      color: C.body,
    });
    addRule(slide, x, y + 0.82, 5.76, C.line, 0.7);
  });
  addBox(
    slide,
    "Evidence boundary: saved weight features plus training scores form predictions. Declared source links evaluate frozen retrieval; collection metadata shaped selection and one holdout. Checkpoint revision remains unverified.",
    0.68,
    6.26,
    11.98,
    0.65,
    {
      fill: C.berkeley,
      line: C.berkeley,
      color: C.white,
      fontSize: 10.6,
      bold: true,
      align: "center",
    }
  );
}

// 22. Proposed scale-up study
{
  const slide = baseSlide(
    "APPENDIX A5 · OCTOBER RESEARCH PLAN",
    "Do weight measurements improve model selection for a task?",
    C.dartmouth,
    22,
    "Source: docs/research/scaleup/model_zoo_frontier_research_plan_2026-10-23.md; proposed experiments and milestones."
  );
  addText(slide, "Measure score lost versus the best feasible model, evaluation cost, and constraint failures.",
    0.68, 1.25, 11.98, 0.32, {fontSize: 12, color: C.body});
  const steps = [
    ["Task + constraints", "examples, memory, latency", C.booking, C.bookingLight],
    ["Retrieve candidates", "use model groups and evidence", C.dartmouth, C.dartmouthLight],
    ["Estimate suitability", "predict scores for the task", C.booking, C.bookingLight],
    ["Evaluate finalists", "spend the remaining budget", "9A6A00", C.goldLight],
  ];
  steps.forEach(([title, detail, color, fill], index) => {
    const x = 0.68 + index * 3.16;
    addBox(slide, title, x, 1.94, 2.45, 0.66, {fill, line: color, color: C.ink,
      fontSize: 12.5, bold: true, align: "center"});
    addText(slide, detail, x, 2.77, 2.45, 0.46, {fontSize: 10.5, color: C.body, align: "center"});
    if (index < steps.length - 1) addArrow(slide, x + 2.58, 2.27, 0.43, color);
  });
  addText(slide, "COLLECTOR CAPABILITIES NOW", 0.68, 3.44, 5.7, 0.24,
    {fontSize: 10, bold: true, color: C.booking});
  addRule(slide, 0.68, 3.82, 5.7, C.booking, 1.6);
  addText(slide, "Pinned revisions, canonical matrix records, full computed spectra, and explicit coverage.\nNew records still need verified benchmark joins and prediction experiments.",
    0.68, 4.02, 5.7, 0.82, {fontSize: 11.5, color: C.body});
  addText(slide, "PROPOSED EVIDENCE COMPARISON", 7.02, 3.44, 5.63, 0.24,
    {fontSize: 10, bold: true, color: C.dartmouth});
  addRule(slide, 7.02, 3.82, 5.63, C.dartmouth, 1.6);
  addText(slide, "Scientific track: weights plus training scores.\nPractical track: documentation, historical scores, and weights.\nTest new roots, tasks, and future releases.",
    7.02, 4.02, 5.63, 0.82, {fontSize: 11.5, color: C.body});
  addText(slide, "PROPOSED COLLECTION MILESTONES", 0.68, 5.16, 11.98, 0.24,
    {fontSize: 10, bold: true, color: C.muted});
  addRule(slide, 0.86, 5.66, 11.57, C.line, 2);
  const milestones = [
    ["1k pinned artifacts", "Measure collection cost and coverage", C.booking],
    ["5k-10k measured", "Compare task choices and evidence", C.dartmouth],
    ["50k measured target", "Release the zoo and benchmark", "9A6A00"],
  ];
  milestones.forEach(([title, detail, color], index) => {
    const x = 0.68 + index * 4.17;
    addDot(slide, x + 0.18, 5.66, 0.12, color);
    addText(slide, title, x, 5.96, 3.64, 0.34, {fontSize: 16, bold: true, color});
    addText(slide, detail, x, 6.44, 3.64, 0.38, {fontSize: 10.5, color: C.body});
  });
}

for (const slide of pptx._slides) {
  warnIfSlideHasOverlaps(slide, pptx, {
    muteContainment: true,
    ignoreLines: true,
    ignoreDecorativeShapes: true,
  });
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

pptx.writeFile({ fileName: path.join(HERE, "model_zoo_htsr_study.pptx") });
