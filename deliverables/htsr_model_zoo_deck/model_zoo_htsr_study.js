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
pptx.subject = "Metadata-isolated HTSR lineage and performance study";
pptx.title = "Model weights as evidence";
pptx.lang = "en-US";
pptx.theme = { headFontFace: DISPLAY, bodyFontFace: FONT, lang: "en-US" };

const counts = DATA.counts;
const coverage = DATA.profile_sources;
const collections = DATA.collections;
const internal = DATA.internal;
const gallery = DATA.gallery;
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

function perf(scope, generation, kind = "knn") {
  return DATA.performance[scope][generation][kind];
}

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
    charSpacing: 1.1,
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
    charSpacing: 1.5,
  });
  addText(slide, "Model weights\nas evidence", 0.72, 1.18, 7.2, 1.52, {
    fontFace: DISPLAY,
    fontSize: 40,
    bold: true,
    color: C.white,
  });
  addText(
    slide,
    "Lineage grouping and performance prediction without descriptive metadata",
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
    f(perf("interpolation", "modern").model_weighted_mae, 2),
    "modern score error",
    5.92,
    4.45,
    2.0,
    "7ACBA5",
    "each model tested once",
    true
  );
  addCoverArtwork(slide);
  addText(slide, `Completed run: ${DATA.run_id}`, 0.75, 6.93, 7.4, 0.18, {
    fontSize: 7.8,
    color: "B9C5D6",
  });
}

// 2. Boundary
{
  const slide = baseSlide(
    "STUDY BOUNDARY",
    "Make predictions first; use descriptive metadata only to check them",
    C.berkeley,
    2,
    `Source: ${DATA.run_id}/metadata.json`
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
      input: "available roles + training scores",
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
    "OPEN ONLY AFTER PREDICTION  |  lineage · family · architecture · model type · parameter count · source · ancestry graph · held-out score",
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
    "Model IDs only join files. The base-candidate list defines the possible answers, as it does in the reference paper.",
    0.66,
    6.15,
    11.65,
    0.42,
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
    "Sources: collection code, completed weight-only run, and post-prediction validation tables."
  );
  const phases = [
    {
      x: 0.62,
      number: "1",
      title: "Assemble model lists",
      detail: "Phase 2 and ablation IDs\ncanonicalize and remove duplicates",
      output: "19,713 distinct models",
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
      detail: "open provenance and held-out scores\nmeasure agreement and error",
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
    "The predictor sees model weights, the fixed candidate-base list, and training benchmark scores. Descriptive metadata is opened only in phase 4.",
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
    "Normalize owner/repository and revision; keep one row per canonical checkpoint.",
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
    "Collection metadata shapes coverage. It is removed before lineage grouping and is not used as a performance feature.",
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
    "Phase 2 is broad; ablation is benchmark-heavy",
    C.dartmouth,
    5,
    "Sources: completed run's Phase 2 and ablation source tables; architecture graph used descriptively after selection."
  );
  addImage(slide, "corpus_types.png", 0.38, 1.17, 6.15, 3.82);
  addImage(slide, "corpus_architectures.png", 6.72, 1.17, 6.15, 3.82);
  addBox(
    slide,
    `PHASE 2 · ${n(collections.phase2.selected)}\n${n(collections.phase2.benchmark_rows)} benchmark scores · ${n(collections.phase2.lineage_metadata_rows)} lineage records\n${n(collections.phase2.architecture_rows)} architecture labels · ${n(collections.phase2.parameter_rows)} parameter counts`,
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
    `ABLATION · ${n(collections.ablation.selected)}\n${n(collections.ablation.benchmark_rows)} benchmark scores · ${n(collections.ablation.lineage_metadata_rows)} lineage records\n${n(collections.ablation.architecture_rows)} architecture labels · ${n(collections.ablation.parameter_rows)} parameter counts`,
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
    "Type and architecture labels describe the datasets only. 'Other labelled' combines lower-count architecture families.",
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
    "Source labels explain coverage here. They are removed before clustering, candidate-base matching, and performance prediction.",
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
    "Stored DKS alpha · depth scaled to [0,1] · 32 positions per observed role · within-role ranks · compare only shared roles",
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
    "Fixed-tail Hill alpha, k=n/2 · role-wise Spearman · z-scored injective order-preserving dynamic programming when depths differ",
    6.88,
    4.48,
    5.78,
    1.03,
    { fontSize: 11, color: C.body }
  );
  addBox(
    slide,
    "Same scientific idea; different alpha estimator and unequal-depth calculation. Numerical results are not an exact replication.",
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
    "Seven roles define the paper-comparison core, not general eligibility",
    C.booking,
    8,
    `Sources: ${DATA.run_id}/population_ledger.csv and metadata.json`
  );
  addImage(slide, "corpus_coverage.png", 0.52, 1.24, 8.05, 5.55);
  addStat(slide, n(counts.strict_seven_role_models), "strict seven-role core", 8.92, 1.37, 2.9, C.berkeley);
  addStat(slide, n(counts.lineage_models), "lineage assignments", 8.92, 2.62, 2.9, C.booking);
  addStat(
    slide,
    n(counts.performance_profiled_models),
    "performance profiles",
    8.92,
    3.87,
    2.7,
    C.dartmouth
  );
  addBox(
    slide,
    `Partial lineage patterns must retain the complete-profile nearest model in the top ten at least 90% of the time. Performance accepts the label-free ${n(counts.minimum_usable_roles)}-role cutoff and requires every neighbor to share that many roles.`,
    8.82,
    5.28,
    3.45,
    0.75,
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
    `HOW 17 WAS CHOSEN\n\nFit candidate counts on\n${n(internal.n_core_models)} complete profiles\n\nKeep the count with\nthe best separation`,
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
    "Compare the 17 weight-derived groups with three independent label sets",
    C.dartmouth,
    10,
    `Sources: ${DATA.run_id}/lineage_validation_metrics.csv and lineage_cluster_assignments.csv`
  );
  addImage(slide, "validation_agreement.png", 0.42, 1.26, 7.28, 4.82);
  addText(slide, "LABELS OPENED AFTER GROUPING", 7.96, 1.37, 4.5, 0.24, {
    fontSize: 10,
    bold: true,
    color: C.dartmouth,
  });
  const labelSets = [
    ["Human-reviewed", `${n(manual.n_models)} models assigned to 11 broad base families by manual review.`, C.bookingLight, C.booking],
    ["Metadata agreement", `${n(consensus.n_models)} models assigned to 11 broad families when metadata rules agree.`, C.dartmouthLight, C.dartmouth],
    ["Graph roots", `${n(graphRoot.n_models)} models traced to ${n(graphRoot.n_families)} individual root checkpoints in the provenance graph.`, C.goldLight, "9A6A00"],
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
    "Match each model against a fixed list of known base checkpoints",
    C.berkeley,
    11,
    `Sources: ${DATA.run_id}/lineage_gallery_metrics.csv; reference paper Sec. 5.2 and 5.4`
  );
  addImage(slide, "gallery_accuracy.png", 0.42, 1.26, 7.55, 5.35);
  addStat(slide, `${n(gallery.n_profiled_roots)} / ${n(gallery.n_supplied_roots)}`, "available / supplied bases", 8.35, 1.42, 1.95, C.berkeley);
  addStat(slide, n(gallery.n_scored_queries), "models scored", 10.62, 1.42, 1.65, C.booking);
  addStat(slide, pct(gallery.micro_top1, 2), "top choice correct", 8.35, 2.91, 1.95, C.dartmouth, "across all models");
  addStat(slide, pct(gallery.root_macro_top1, 2), "top choice correct", 10.62, 2.91, 1.72, "9A6A00", "each base weighted equally");
  addBox(
    slide,
    "The possible answers are fixed before prediction. A model is compared with all 29 available base profiles and the bases are ranked by similarity.",
    8.35,
    4.36,
    4.02,
    1.02,
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
    "A base outside this candidate list cannot be selected. Provenance is opened afterward only to score the ranking.",
    8.35,
    5.68,
    4.02,
    0.72,
    {
      fill: C.riskLight,
      line: C.risk,
      color: C.risk,
      fontSize: 10.2,
      bold: true,
      align: "center",
    }
  );
}

// 12. Paper comparison
{
  const slide = baseSlide(
    "PAPER COMPARISON",
    "The evaluation boundary matches; the estimator and scale do not",
    C.gold,
    12,
    "Paper: arXiv:2607.03377v1. Current metrics: lineage_gallery_metrics.csv."
  );
  const rows = [
    ["Dimension", "Reference paper", "Current study"],
    ["Model selection", "Public provenance", "Fixed candidate list + labels opened afterward"],
    ["Queries / bases", "128 / 8", `${n(gallery.n_scored_queries)} / ${n(gallery.n_profiled_roots)} profiled`],
    ["Prediction input", "Weights only", "Weights only"],
    ["Alpha", "Fixed-tail Hill", "Stored DKS"],
    ["Unequal depth", "Z-score + order-preserving dynamic programming", "32-point normalized-depth interpolation"],
    ["Overall top-1", "98.44%", pct(gallery.micro_top1, 2)],
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
    "Do not interpret the accuracy gap as an estimator ranking: the candidate sets, labels, coverage, alpha fit, and depth handling differ.",
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
    "Every scored model is tested using models outside its test part",
    C.booking,
    13,
    `Sources: ${DATA.run_id}/performance_frozen_splits.csv and performance_prediction_freeze.csv`
  );
  addText(slide, "BUILD FIVE TEST PARTS", 0.72, 1.34, 4.2, 0.26, {
    fontSize: 10,
    bold: true,
    color: C.booking,
  });
  const setupSteps = [
    ["Group identical\nobserved-role summaries", C.bookingLight, C.booking],
    ["Divide those groups\ninto five parts", C.surface, C.muted],
    ["Save the division\nbefore prediction", C.dartmouthLight, C.dartmouth],
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
    ["Find 3 closest\ntraining models", C.bookingLight, C.booking],
    ["Predict a weighted\nscore average", C.surface, C.muted],
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
  addBox(slide, "A fold is one of the five parts. Each part takes one turn as the held-out test set.", 0.76, 5.25, 5.75, 0.66, {
    fill: C.bookingLight,
    line: C.booking,
    color: C.ink,
    fontSize: 10.3,
    bold: true,
    align: "center",
  });
  addBox(slide, "Identical summaries stay together. Closer spectral matches receive more weight.", 6.8, 5.25, 5.75, 0.66, {
    fill: C.dartmouthLight,
    line: C.dartmouth,
    color: C.ink,
    fontSize: 10.3,
    bold: true,
    align: "center",
  });
  addText(
    slide,
    "Modern and legacy benchmark sets are trained and evaluated separately. Descriptive metadata is never a predictor input.",
    0.72,
    6.31,
    11.9,
    0.3,
    { fontSize: 10.2, color: C.muted, align: "center" }
  );
}

// 14. Interpolation
{
  const slide = baseSlide(
    "PERFORMANCE RESULTS",
    "Nearest spectral neighbors predict well when similar models are represented",
    C.booking,
    14,
    `Source: ${DATA.run_id}/performance_metrics.csv`
  );
  addImage(slide, "performance_interpolation.png", 0.48, 1.22, 8.05, 5.62);
  const modern = perf("interpolation", "modern");
  const legacy = perf("interpolation", "legacy");
  addText(slide, "Modern", 8.8, 1.39, 1.7, 0.3, {
    fontSize: 15,
    bold: true,
    color: C.booking,
  });
  addStat(slide, f(modern.model_weighted_mae, 3), "MAE", 8.8, 1.86, 1.28, C.booking);
  addStat(slide, f(modern.spearman, 3), "Spearman", 10.37, 1.86, 1.45, C.dartmouth);
  addText(slide, `${n(modern.n_models)} models / ${n(modern.n_profile_groups)} unique spectral summaries`, 8.8, 3.04, 3.45, 0.28, {
    fontSize: 9.3,
    color: C.muted,
  });
  addText(slide, "Legacy", 8.8, 3.62, 1.7, 0.3, {
    fontSize: 15,
    bold: true,
    color: C.dartmouth,
  });
  addStat(slide, f(legacy.model_weighted_mae, 3), "MAE", 8.8, 4.09, 1.28, C.booking);
  addStat(slide, f(legacy.spearman, 3), "Spearman", 10.37, 4.09, 1.45, C.dartmouth);
  addText(slide, `${n(legacy.n_models)} models / ${n(legacy.n_profile_groups)} unique spectral summaries`, 8.8, 5.27, 3.45, 0.28, {
    fontSize: 9.3,
    color: C.muted,
  });
  addBox(
    slide,
    "These results cover models with similar weight summaries in training. The next test removes complete weight groups.",
    8.8,
    5.83,
    3.42,
    0.67,
    {
      fill: C.goldLight,
      line: C.gold,
      color: C.ink,
      bold: true,
      align: "center",
      fontSize: 10.1,
    }
  );
}

// 15. Stress test
{
  const slide = baseSlide(
    "NEW-REGION STRESS TEST",
    "Holding out a model's entire weight group reveals the limit",
    C.risk,
    15,
    `Sources: ${DATA.run_id}/performance_cluster_holdout_metrics.csv and performance_cluster_holdout_predictions.csv`
  );
  addImage(slide, "performance_stress.png", 0.44, 1.2, 7.2, 5.72);
  addImage(slide, "modern_prediction_scatter.png", 7.75, 1.2, 5.05, 3.15);
  const modern = perf("cluster_holdout", "modern");
  const legacy = perf("cluster_holdout", "legacy");
  addBox(
    slide,
    `MODERN TASKS\nMAE ${f(modern.model_weighted_mae, 3)} · Spearman ${f(modern.spearman, 3)}\ntraining average ${f(perf("cluster_holdout", "modern", "baseline").model_weighted_mae, 3)}`,
    8.05,
    4.54,
    2.05,
    1.18,
    {
      fill: C.bookingLight,
      line: C.booking,
      color: C.ink,
      fontSize: 10.4,
      bold: true,
      align: "center",
    }
  );
  addBox(
    slide,
    `LEGACY TASKS\nMAE ${f(legacy.model_weighted_mae, 3)} · Spearman ${f(legacy.spearman, 3)}\ntraining average ${f(perf("cluster_holdout", "legacy", "baseline").model_weighted_mae, 3)}`,
    10.34,
    4.54,
    2.05,
    1.18,
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
    "For legacy tasks, this is worse than using the training-set average. Similarity must be checked before trusting a score estimate.",
    8.02,
    6.05,
    4.4,
    0.6,
    { fontSize: 10.2, color: C.risk, bold: true, align: "center" }
  );
}

// 16. Decision
{
  const slide = baseSlide(
    "CONCLUSION",
    "What the results show, and where uncertainty remains",
    C.berkeley,
    16,
    "Summary of the frozen lineage and performance evaluations in the completed run."
  );
  const columns = [
    {
      x: 0.62,
      title: "Observed",
      color: C.dartmouth,
      fill: C.dartmouthLight,
      items: [
        "17 repeatable weight-derived groups",
        "Known base appears in top 3 for 98.4%",
        `Modern-score error is ${f(perf("interpolation", "modern").model_weighted_mae, 3)} in the ordinary test`,
      ],
    },
    {
      x: 4.48,
      title: "Interpretation",
      color: C.booking,
      fill: C.bookingLight,
      items: [
        "Alpha order through depth contains family information",
        "Broad families are easier than exact checkpoint ancestry",
        "Score estimates depend on nearby training examples",
      ],
    },
    {
      x: 8.34,
      title: "Remaining uncertainty",
      color: "9A6A00",
      fill: C.goldLight,
      items: [
        "Similarity cannot establish parent or training direction",
        "A base absent from the candidate list cannot be returned",
        "Legacy errors exceed the mean baseline when a whole group is unseen",
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
    "Overall: weight-derived alpha summaries support broad grouping, known-base matching, and local score estimation. Exact ancestry needs an additional direct weight-matrix check.",
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
    charSpacing: 1.5,
  });
  addText(slide, "Methods and definitions", 0.72, 1.27, 8.7, 0.78, {
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
    "Sources: Model-Zoo/esd_experiment/gputracker and Model-Zoo/docs/operations/analysis.md"
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
    "Each weight matrix becomes a fitted spectrum and a structured record",
    C.dartmouth,
    19,
    "Sources: Model-Zoo/esd_experiment/src/worker.py and Model-Zoo/README.md"
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
    ["Optional eigenvalues", "larger files; useful for later re-fitting"],
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
    "Analysis continues past inaccessible or repeatedly failing models after the configured retry limit. Completed records remain available for aggregation.",
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
    "Primary sources: arXiv:2608.07786, arXiv:2608.14929, arXiv:2510.06738."
  );
  addBox(
    slide,
    `STAGE 1\nHTSR coarse retrieval\n${n(counts.lineage_models)} assigned models`,
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
    "OUTPUT\nrelated / unrelated evidence\nwith calibrated thresholds",
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
      note: "Coarse family separation plus fine directional geometry.",
      color: C.booking,
    },
    {
      x: 4.48,
      id: "arXiv:2608.14929",
      title: "Centered residual signatures",
      note: "Removes shared identity structure before comparing residual blocks.",
      color: C.dartmouth,
    },
    {
      x: 8.24,
      id: "AWM · arXiv:2510.06738",
      title: "Assignment + centered alignment",
      note: "Aligns matrices, then compares their centered weight structure.",
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
    addText(slide, paper.note, paper.x, 4.69, 3.38, 0.68, {
      fontSize: 10,
      color: C.body,
      align: "center",
    });
  });
  addBox(
    slide,
    "Not tested here: the saved cache has scalar eigenvalue summaries, not singular vectors or centered weight matrices.",
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
    ["Five-part test", "Divide models into five fixed parts; test each part using the other four."],
    ["Known-base test", "Choose the closest model from a fixed list of possible base checkpoints."],
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
    "Evidence boundary: model weights and benchmark training scores create predictions. Provenance and family metadata are opened only after predictions are saved, to measure correctness.",
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

for (const slide of pptx._slides) {
  warnIfSlideHasOverlaps(slide, pptx, {
    muteContainment: true,
    ignoreLines: true,
    ignoreDecorativeShapes: true,
  });
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

pptx.writeFile({ fileName: path.join(HERE, "model_zoo_htsr_study.pptx") });
