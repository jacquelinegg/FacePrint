# FACEPRINT — AI Identity QA for Advertising

**Know how much of a real person survives AI generation.**

An agency uploads the original photo of a model and gets an **Identity Lock** (the baseline). Every
AI-generated advertising visual is then compared against it: **how much the face still resembles the
hired person** (the verdict) and **exactly which characteristics were lost or altered** (the
explanation, with measured numbers, not adjectives).

The tool is an **independent QA layer**: it does not generate advertising itself — it audits whatever
the generator (YouCam, Firefly, Midjourney, in-house diffusion, …) produced, and it can drive one
real generator (YouCam) end-to-end to prove the loop works on live output.

---

## Table of contents

1. [The problem](#1-the-problem)
2. [What it does](#2-what-it-does)
3. [Core concepts](#3-core-concepts)
4. [Use cases](#4-use-cases)
5. [Architecture and logic distribution](#5-architecture-and-logic-distribution)
6. [The measurement model in detail](#6-the-measurement-model-in-detail)
7. [Identity boundary calibration](#7-identity-boundary-calibration)
8. [Identity Lock — the closed loop](#8-identity-lock--the-closed-loop)
9. [YouCam integration layer](#9-youcam-integration-layer)
10. [Stress test](#10-stress-test)
11. [Talent policy and the PDF campaign report](#11-talent-policy-and-the-pdf-campaign-report)
12. [Web UI](#12-web-ui)
13. [CLI reference](#13-cli-reference)
14. [Technology stack](#14-technology-stack)
15. [Configuration and secrets](#15-configuration-and-secrets)
16. [Tests](#16-tests)
17. [Evidence and proof](#17-evidence-and-proof)
18. [Demo playbook](#18-demo-playbook)
19. [Honest limitations](#19-honest-limitations)
20. [Integrating another generator](#20-integrating-another-generator)
21. [Project structure](#21-project-structure)
22. [Ethics, consent and legal position](#22-ethics-consent-and-legal-position)
23. [Quick start](#23-quick-start)

---

## 1. The problem

A brand pays for a *face* — a specific model, with a specific look, usually under a contract that
says the campaign may not alter them beyond agreed limits. Then production generates 30–200 AI
variants of that face for different formats, and something goes wrong quietly:

* the face becomes *slightly* slimmer, the eyes a bit larger, the jaw narrower — each edit small
  enough to pass a glance, all of them together turning the model into somebody else;
* beauty filters smooth the skin so hard the person's texture (and apparent age) disappears;
* a hair/makeup effect reshapes the cheeks and brows;
* a head turns or an expression changes, so the shot no longer matches the approved one.

Today this is checked by eye ("looks fine to me"), which is slow, unrecordable and unprovable.
A cosine-similarity-only tool is equally weak in the other direction: cosmetic edits keep ArcFace
cosine **high** (the person is still recognisable) while the face has visibly changed — the exact
case a single-number tool misses.

FACEPRINT answers both questions per image, in a repeatable, calibrated and reportable way.

---

## 2. What it does

| Layer | Question it answers | How |
|---|---|---|
| **Identity score (verdict)** | "Is this still the same person?" | ArcFace embeddings (InsightFace `buffalo_l`), cosine similarity, thresholds calibrated on LFW → `● SAFE / ◐ REVIEW / ✕ IDENTITY DRIFT` |
| **Attribute breakdown (explanation)** | "What exactly changed?" | 8 scored groups (face shape, eyes, brows, lips, nose, skin tone, expression, head pose), each from 2 independent signals: 50 % landmark geometry + 50 % pixel-level region comparison. Skin tone carries **two** signals (colour ΔE and texture), and the category drops if **either** moves. Every measure is reported as `reference → variant`, magnitude and score. |
| **Identity boundary** | "Where is the line?" | SAFE / REVIEW / DRIFT thresholds derived from data (median of genuine pairs, 99th percentile of impostor pairs), with the histograms shipped in the repo |
| **Stress test** | "Where does drift begin?" | 10 controlled local mutations × 11 strengths; two measured layers per step — identity cosine and the weakest attribute group — plus the exact strength at which each breaks |
| **Identity Lock (closed loop)** | "How strong can the AI effect be before the person is gone?" | Real generative effects (YouCam aging / hair / makeup / smile / skin), then the strongest version that keeps the person SAFE — by blending where the API has no strength dial, and by the API's own strength parameter where it has one |
| **Lock recommendations** | "What should the art director lock?" | Derived from the lost/changed categories, with the measured values that triggered them |
| **YouCam attribute analysis** | "What does a commercial face-analysis API see?" | Named attributes (face shape, cheekbones, eye shape/angle/size/setting, brows, lips, nose, ratios, age/sex) compared before → after, e.g. `faceshape: Square → Oval` |
| **Policy + PDF report** | "Is this campaign compliant, and can I show the client?" | Per-talent rules (minimum similarity + locked attribute groups) → `COMPLIANT / REVIEW / NON-COMPLIANT` per image, with a tamper-evident Report ID (SHA-256 over the images and results) |

Everything that can be run offline runs offline; only the YouCam features and the LFW calibration
need the network.

---

## 3. Core concepts

**Identity Lock (baseline).** The reference photo of the model is embedded once: ArcFace embedding +
106 landmarks + 5 keypoints + the full attribute measurement vector. Every later image is compared
against *this* — never against itself, and never against "the largest face in the frame".

**Verdict statuses.**

| Glyph | Status | Meaning |
|---|---|---|
| `●` | `SAFE` | cosine ≥ **0.67** (calibrated) — recognisably the same person |
| `◐` | `REVIEW` | cosine ≥ **0.15** (calibrated) — changed; read the attribute breakdown |
| `✕` | `IDENTITY DRIFT` | cosine < 0.15 — a different face |
| `○` | `NO FACE` | no face detected in the image (scored, never guessed) |

**Attribute groups (8).** Face shape, Eyes, Brows, Lips, Nose, Skin tone, Expression, Head pose.
Apparent age and predicted sex are *reported as information only* — the age model's own error
(~±5 years) means the score stays pinned at 100 and carries no signal.

**Preservation bands.** Per group: `≥ 80` preserved · `60–79` changed · `< 60` lost.

**Two independent signal families, deliberately.** ArcFace cosine answers "same person?", the
attribute layer answers "what changed?". A cosmetic edit (lipstick, tone grade) barely moves cosine
but wrecks the attribute layer; a heavy stylisation can keep group geometry while cosine collapses.
Both curves are shown side by side everywhere the tool reports.

---

## 4. Use cases

### 4.1 Campaign QA — "which of these 40 AI visuals can ship?"
Upload the original model photo once, then every AI-generated variant (from any generator). Each
image gets a verdict + the 8 group scores + preserved/changed/lost lists + concrete
`reference → variant` rows (`face_ratio 0.78 → 0.71`, `skin_lab L*64 a*+18 b*+26 → L*70 a*+14 b*+22`,
`lipstick became redder`…). The table is exported as CSV (`faceprint_qa.csv`) for the producer.

### 4.2 Pre-flight check of a generator's settings
Before rendering the whole campaign, the studio generates one test image per planned setting
(lighting grade, beauty level, eye size, aspect crop) and checks which settings break the identity
boundary. Cost of one test image instead of 200 wrong finals.

### 4.3 Identity Lock — "how far can we push the effect on this specific model?"
Run a real YouCam effect (aging, hair-transfer, makeup, ai-smile, skin-simulation) on the reference
photo, then sweep strength and report the **maximum SAFE intensity** — e.g. the aging effect stays
SAFE up to 70 % blend on the demo model, hair-transfer stays SAFE at 100 %. Output: the identity
curve, the full-effect image, the identity-locked image, and the numbers as a CSV.

### 4.4 Contract / likeness review (policy mode)
Encode the talent's rules as a policy file: minimum similarity + locked attribute groups with
minimum preservation (`{"Face shape": 70, "Eyes": 70}`). Every image gets `COMPLIANT / REVIEW /
NON-COMPLIANT` with the specific violations, and the whole set is exported as a branded PDF with a
Report ID — the artefact that goes to the client or the talent's agent.

### 4.5 Regression harness for a pipeline that generates faces
The stress test is a deterministic benchmark: 10 mutations × 11 strengths, two layers measured per
step, exported as CSV. If a detector, landmarker or generator version changes, re-run the sweep and
diff the breakpoints (`review_at`, `drift_at`, `attr_at` per mutation).

### 4.6 Validation of a *different* face-analysis stack
The YouCam layer is a full, live-tested client (File API → task → poll → attribute diff). It shows
what a commercial named-attribute API concludes about the same pair of images, which both
cross-checks the local heuristics and demonstrates the integration pattern for any other S2S API.

---

## 5. Architecture and logic distribution

### 5.1 Data flow

```
 reference photo ──► FaceEngine (InsightFace buffalo_l: detect + ArcFace + 106 landmarks + 5 kps + age/gender)
                          │
                          ├─► Baseline .................................. Identity Lock (embedding + measured attributes)
                          │        │
   AI variant (upload /────┤        ├─► FaceEngine.analyze(img, reference=…)  → matching face (not the largest)
   YouCam generate /───────┘        │
   local mutation)                  ├─► attributes.measure()  → same measurement vector for the variant
                                   ├─► cosine(reference, variant) ──► status_of() with calibrated thresholds → SAFE / REVIEW / DRIFT
                                   ├─► attributes.compare() ──► 8 group scores (geometry ⊕ appearance)
                                   └─► attributes.explain() ──► per-attribute rows: reference → variant, Δ, score, note
                                                       │
   YouCam generative effect ─► generators.generate() ─► lock.find_safe_intensity() / find_safe_native()
                                                       │        (blend sweep or native strength sweep → max SAFE intensity)
                                                       ▼
   Policy (rules) ─► policy.evaluate() ─► report.build() ─► PDF (verdicts, breakdowns, YouCam diffs, Report ID)
                                                       │
   UI (Streamlit, 6 tabs) ◄────────────────────────────┴──► CSV / PDF downloads
                      stress_sweep() ◄── transforms.py (10 local mutations × 11 strengths)
                      calibrate.run() ◄── labelled dataset <root>/<person>/*.jpg (e.g. LFW) → calibration.json
```

### 5.2 Module map — who owns what

| Module | Responsibility | Key entry points |
|---|---|---|
| `app.py` | Streamlit application: 6 tabs, uploaders, session state, wiring everything together | `engine()`, `Baseline`, `base.check()`, `stress_sweep()`, `generate()`, `find_safe_*()`, `evaluate()`, `build()` |
| `faceprint/engine.py` | The only face-model wrapper. Loads `buffalo_l` once (singleton), exposes detection, ArcFace embedding, 106 landmarks, 5 keypoints, age/gender. Multi-face handling: largest *or* best match against a reference embedding | `FaceEngine.analyze_all()`, `.analyze(img, reference=None)`, `cosine()` (`engine.py:39`) |
| `faceprint/qa.py` | Identity QA orchestration: threshold loading, status mapping, the `Baseline` (Identity Lock) and `check()`, the stress sweep | `thresholds()`, `status_of()` (`qa.py:14`), `Baseline.check()` (`qa.py:38`), `stress_sweep()` (`qa.py:62`) |
| `faceprint/attributes.py` | The interpretable measurement layer: 13 geometric ratios + expression/pose proxies + skin colour/texture + lip colour + canonical-frame appearance NCC; scoring and the human-readable diff | `measure()` (`attributes.py:136`), `compare()` (`attributes.py:177`), `appearance()` (`attributes.py:168`), `explain()` (`attributes.py:261`) |
| `faceprint/transforms.py` | 10 deterministic local mutations (strength 0–1) that stand in for generative edits in the stress test | `TRANSFORMS`, `apply()` (`transforms.py:96`) |
| `faceprint/calibrate.py` | Empirical threshold calibration on a labelled folder: genuine vs impostor cosine distributions → safe/review thresholds + histograms | `run()` (`calibrate.py:9`) |
| `faceprint/youcam.py` | YouCam (Perfect Corp) S2S client: File API (presigned PUT), task POST + poll, image download, hair templates; plus the named-attribute comparison spec | `YouCamClient.upload()` (`youcam.py:110`), `.analyze*()` (`youcam.py:121`), `.run_generative()` (`youcam.py:161`), `.templates()`, `compare_results()` (`youcam.py:226`) |
| `faceprint/generators.py` | Effect → endpoint table, per-endpoint file specs, face framing (`_fit`), compositing back into the original frame (`_composite`), one `generate()` for all effects | `generate()` (`generators.py:77`), `EFFECTS`, `NATIVE` |
| `faceprint/lock.py` | The Identity Lock solver: blend sweep (α 0→1) and native-strength sweep; returns curve + max SAFE/REVIEW intensity + images | `find_safe_intensity()` (`lock.py:10`), `find_safe_native()` (`lock.py:27`) |
| `faceprint/policy.py` | Talent policy rules and the compliance verdict | `Policy` (`policy.py:7`), `evaluate()` (`policy.py:21`) |
| `faceprint/report.py` | PDF campaign report (ReportLab): summary, per-image blocks, group scores, violations, measured changes, YouCam diffs, tamper-evident Report ID | `build()` (`report.py:29`) |
| `faceprint/ui.py` | The visual system: graphite/bronze/bone theme, status chips (glyph + density, **no traffic-light hues**), Vega-Lite chart specs with a metallic ramp, HTML legends | `inject()`, `wordmark()`, `chip()`, `vega_line()` (`ui.py:399`), `vega_bar()` (`ui.py:421`) |
| `faceprint/cli.py` | Argparse CLI: `qa`, `stress`, `calibrate`, `youcam`, `lock`, `campaign` | `main()` (`cli.py:6`) |
| `tests/` | 9 test files (offline where possible, live YouCam mocked) | see [Tests](#16-tests) |
| `scripts/` | Operator tools: live proof, calibration data, human study, demo-data builder | see [Evidence](#17-evidence-and-proof) |

### 5.3 Where each decision lives (single responsibility)

* **Face maths** (detection, embeddings, landmarks, multi-face choice) — only in `engine.py`.
  Nothing else touches InsightFace.
* **"Is it the same person?"** — only in `qa.py` (`cosine` + `status_of` + thresholds from
  `calibration.json`).
* **"What changed, by how much, in what units?"** — only in `attributes.py` (`measure` produces a
  fixed-shape measurement dict; `compare` turns two dicts into scores; `explain` turns them into
  human rows). UI, CLI, PDF and stress test all consume the same three functions — one measurement
  vocabulary everywhere.
* **"What do we call it?"** — statuses and bands are defined once in `qa.py`; policy verdicts once in
  `policy.py`; the UI never redefines them.
* **Network / third-party API** — only in `youcam.py` and `generators.py`. The rest of the codebase
  is network-free and testable offline.
* **Presentation** — theme and charts only in `ui.py`; `app.py` only composes layout.

---

## 6. The measurement model in detail

### 6.1 Identity score

ArcFace (via InsightFace `buffalo_l`, 512-d L2-normalised embedding) → cosine similarity ∈ [−1, 1].
The UI shows `100 · max(cosine, 0)` as a percentage. Thresholds come from
`faceprint/calibration.json` and are **calibrated on LFW** (see §7).

With more than one face in the frame, the compared face is the one **most similar to the reference
embedding** — so a second person can neither hide the real match nor fake one (`engine.py:39`,
`engine.py:45`). The advice row states that the best match was used.

### 6.2 The eight attribute groups — two signals each

Every group is scored from two *independent* signal families, averaged 50/50:

1. **Geometry** — ratios from the 106-point 2D landmark set. The landmark cloud is first aligned
   (`_aligned`, `attributes.py:43`) so ratios are scale- and rotation-normalised; most ratios are
   additionally divided by the inter-ocular distance. 13 geometric attributes
   (`SPEC`, `attributes.py:14`):

   | Attribute | Category | Tolerance ("clearly noticeable change") |
   |---|---|---|
   | `face_ratio` (w/h) | Face shape | 0.08 rel |
   | `jaw_width` / face width | Face shape | 0.06 rel |
   | `eye_size`, `eye_open`, `eye_spacing` | Eyes | 0.08 / 0.15 / 0.08 rel |
   | `brow_len`, `brow_gap`, `brow_slope` | Brows | 0.15 / 0.20 rel, 6° abs |
   | `lip_width`, `lip_full` | Lips | 0.12 / 0.25 rel |
   | `nose_width`, `nose_len` | Nose | 0.12 / 0.12 rel |

   plus 4 extra signals: `smile`, `mouth_open` (Expression), `yaw`, `pitch` (Head pose) —
   tolerances ~5× the measured noise floor (JPEG / resize / 6° rotation).

2. **Appearance** — the face is warped into a **canonical 256×256 frame** pinned by the two eye
   centres and the mouth centre (`canonical()`, `attributes.py:85`), so region boxes land at fixed
   anatomical positions. Each region is compared with a **local-contrast-normalised cross-correlation
   (NCC)** on a texture map (high-pass) and, for the face shape, on an edge map of the lower face
   (`appearance()`, `attributes.py:168`). This is what catches changes the landmark model simply
   *follows* — enlarged eyes, a slimmed jaw, a filled lip — because the pixels are compared, not the
   keypoints.

   Regions: Eyes (2 boxes), Brows (2 boxes), Nose, Lips, Face shape (jaw contour, y ≥ 150).

Scoring per attribute: `score = 100 · exp(−0.5 · d / tol)` — so a change of one tolerance lands at
≈ 61, two tolerances at ≈ 14. Appearance NCC is mapped `NCC ≤ 0.5 → 0`, `NCC = 1 → 100`.

Groups with a region (Face shape, Eyes, Brows, Nose, Lips) blend 50/50 geometry and appearance;
**Expression** is judged on its geometric signals *and* on how the mouth region looks (50/50 with
the Lips region). **Skin tone** is special: colour (CIELAB ΔE, tol 10) **and** texture (relative
change of high-frequency energy, tol 0.45) — when both are measurable the group is `min(both)`,
because averaging would hide one of the two. A beauty filter that only smooths, or a grade that only
shifts colour, therefore still drops the category.

### 6.3 Colour measurements

* **Skin tone** — median CIELAB of cheek patches; ΔE to the reference (tol 10, "above ~2 is visible").
* **Lip colour** — median CIELAB of the **ring** between two erosions of the lip hull
  (`_lip_lab`, `attributes.py:109`), so an open mouth's teeth/tongue never become the measurement.
  Erosion is relative to **lip size, not image size** (a bug found against real 512–1255 px YouCam
  outputs is regression-tested in `tests/test_lips.py`).

**Not-measurable is not "black".** If a region has no measurable pixels (mask under 40 px, near-black
median, lips out of frame) the value is `None` and the attribute is **skipped**, never scored as
zero — a rule enforced by `tests/test_ui.py`.

### 6.4 Expression and head pose — honest 2-D proxies

`smile` = mouth-corner lift relative to the mouth centre; `mouth_open` = mouth height / inter-ocular
distance; `yaw` = nose offset from the face midline; `pitch` = nose height between eye and mouth
lines. There is **no 3-D pose model** — these are 2-D landmark proxies and are labelled as such
everywhere.

### 6.5 Age and sex

InsightFace `genderage` values are reported as information rows in the full breakdown
("model estimate only — reported, not scored"). The age row even says why: the model's own error is
~±5 years.

### 6.6 The explanation table

For every measured attribute there is a row: **reference value, variant value, magnitude, 0–100
score, plain-language note**, sorted worst-first (`explain()`, `attributes.py:261`). Colour rows
carry CIELAB triples and a human direction ("the skin became warmer", "the lipstick became redder");
ratio rows carry the normalised numbers; NCC rows show "region pixel match; 1.00 = identical". This
is the "what exactly did the AI change" table that sits behind every score — in the UI, in the CSV,
and in the PDF.

---

## 7. Identity boundary calibration

The verdict thresholds are **not** hand-picked. `faceprint/calibrate.py` walks a labelled folder
`<root>/<person>/*.jpg`, embeds every face, and builds:

* **genuine pairs** — all same-person image pairs: **2 923** pairs from **200** LFW identities;
* **impostor pairs** — random different-person pairs: **5 000** pairs.

Derived thresholds (`faceprint/calibration.json`, shipped in the repo):

| Threshold | Value | Derivation | Operating meaning |
|---|---|---|---|
| `safe` | **0.6723** | median of the genuine distribution | "recognisably the same person" — half of genuine photo pairs of the same person score above this |
| `review` | **0.1468** | 99th percentile of the impostor distribution | false-accept rate ≈ 1 % at this boundary; below it, "a different face" |

(The pre-calibration defaults were uncalibrated guesses 0.60 / 0.35 — they remain in the code as
`DEFAULT` for the un-calibrated case and are labelled as such.)

The **Calibration** tab renders both histograms (same-person vs different-people) so the thresholds
are auditable in the UI; the raw histograms are in `calibration.json`.

Reproduce:

```bash
python scripts/get_lfw.py --people 300 --out data/lfw   # downloads LFW (~173 MB), <person>/*.jpg
python -m faceprint calibrate data/lfw [--max-people 400]
```

Any dataset in `<root>/<person>/*.jpg` layout works (own model portfolio, screened studio set).

---

## 8. Identity Lock — the closed loop

`faceprint/generators.py` runs real YouCam generative effects; `faceprint/lock.py` answers the
question the art director actually has: **the strongest version of this effect that still keeps this
person recognisable.**

Two dials, chosen honestly per effect:

* **Blend dial** (`find_safe_intensity`, `lock.py:10`) — for effects with no strength parameter
  (aging, hair, makeup, smile): the generated crop is composited back into the original frame
  (`_composite`, `generators.py:66`) so the blend happens in **original pixels**, then α sweeps 0→1
  in 11 steps; identity is measured at each step. Result: curve, `*_full.jpg`, `*_safe.jpg`, and the
  largest **contiguous** α that stays SAFE (and the largest that stays non-DRIFT).
* **Native dial** (`find_safe_native`, `lock.py:27`) — for `skin-simulation`, which exposes a real
  per-concern 0–1 intensity: one API call per level `(0, 0.1, 0.25, 0.5, 0.75, 1.0)` — denser near
  the low end, where the SAFE boundary typically sits.
* **Aging** has a third, genuine scale: one call returns the whole `res_age` series **12→70**; the
  CLI/UI pick the exact target age (`--age`, below the person's age rejuvenates), and
  `client.last_age` reports what the API actually produced. The UI shows per-age identity for the
  entire series.

Live results on the bundled demo model (see `evidence/20261010_152254/`):

| Effect | Full-effect cosine | Status | Max SAFE intensity |
|---|---|---|---|
| aging (strongest) | 0.571 | ◐ REVIEW | **0.70** |
| hair-transfer | 0.868 | ● SAFE | **1.00** |
| ai-smile | 0.803 | ● SAFE | **1.00** |

---

## 9. YouCam integration layer

`faceprint/youcam.py` is a complete, live-tested client for the Perfect Corp YouCam S2S APIs —
request shapes taken from the official documentation (`docs.perfectcorp.com`, flat
`BasicRunTaskV2` bodies) **and verified against the live API**.

**Face Attribute Analysis** — `POST /s2s/v2.0/task/face-attr-analysis` with 26 flat `features`
(gender, age, faceShape, cheekbones, horizontalThird, verticalFifth, faceAspectRatio, eyeShape,
eyeSize, eyeAngle, eyeDistance, eyelid, eyeAspectRatio, eyeHeightToEyebrowDistance, eyebrowShape,
eyebrowThickness, eyebrowDistance, eyebrowShortness, eyebrowArch, lipShape, upperLipToLowerLip,
noseWidth, noseLength, noseAspectRatio, noseWidthToMouthWidth, noseToLipToChin) and
`face_angle_strictness_level` (default `low` — real-world and AI shots are rarely perfectly level;
the bundled demo face needs `low`).

`compare_results()` (`youcam.py:226`) scores the **named** attributes between reference and variant —
categorical labels as 0/100 per changed label, ratios with relative tolerance, percentage triples
with absolute tolerance, age within ±5 — and lists every concrete change
(`faceshape: Square → Oval`, `Brows / left shape: Soft Angled → Hard Angled`,
`Age / gender: 22 → 46`, …).

**Generative effects** — one `generate()` entry point over five endpoints:

| Effect | Endpoint | Native strength? | File spec (max long / min short px) |
|---|---|---|---|
| `aging` | `POST /s2s/v2.0/task/aging` | `res_age` series 12→70 | 4096 / 320 |
| `hair` | `POST /s2s/v2.1/task/hair-transfer` | — (blend) | 1024 / 480 |
| `makeup` | `POST /s2s/v2.0/task/mu-transfer` | — (blend) | 1024 / 480 |
| `smile` | `POST /s2s/v2.0/task/ai-smile` | — (blend) | 4096 / 480 |
| `skin` | `POST /s2s/v2.0/task/skin-simulation` | per-concern 0–1 (10 concerns) | 2560 / 480 |

Local files go through the **File API**: `POST /s2s/v2.0/file` → presigned `PUT` (no Bearer) →
`file_id`; public URLs are passed through as `*_file_url`. Before upload every image is
auto-framed to the endpoint's spec — cropped to the detected face (30 %/50 % padding, face > 60 % of
frame), roll straightened via the 5 keypoints, scaled into the pixel limits — because out-of-spec
framing fails server-side and wastes billable units. Requirement errors are surfaced verbatim
(`error_face_not_forward_facing`, `error_large_face_angle`, `error_inappropriate_ref_case01`).
Hair templates are browsable through `GET /s2s/v2.1/task/template/hair-transfer` (not billable).

Cost note: Face Analyzer is billed per call (30 units), which is why it is a **separate tab** from
Campaign QA and never runs implicitly.

---

## 10. Stress test

`stress_sweep()` (`qa.py:62`) applies each of the 10 local mutations at 11 strengths (0.0–1.0) and
records, per step: cosine, status, the worst attribute group and its score, and the count of changed
groups — then compresses the sweep into breakpoints per mutation:

* `review_at` — first strength where identity leaves SAFE,
* `drift_at` — first strength where identity DRIFTs (or no face is found),
* `attr_at` — first strength where any attribute group drops below 80.

Two curves are plotted side by side (identity cosine vs worst attribute score) because they
disagree in the interesting cases: `Tone / lighting` burns the Skin-tone group down to 38 while
cosine stays at 0.92 SAFE; `Bigger eyes` breaks identity at 0.7 while degrading attributes from 0.1.
The bundled run (`demo_data/stress_test.csv`) ships these curves for all 10 mutations.

The 10 mutations (`transforms.py`): Skin smoothing, Tone / lighting, Bigger eyes, Fuller lips,
Slimmer jaw, Smile, Head turn, Lipstick, Stylize, Combined beauty (the first five chained).
They are coarse 2-D approximations on purpose — they are a **stand-in** for what a generative tool
might do and a deterministic regression harness, not a claim about any specific generator.

---

## 11. Talent policy and the PDF campaign report

**Policy** (`policy.py`): per-talent / per-campaign rules — `min_cosine` (defaults to the calibrated
SAFE threshold) and `locked` attribute groups with a minimum preservation score. JSON-serialisable
(`Policy.load/save`), see `samples/policy.json` and `demo_data/policy.json`.

`evaluate()` returns `COMPLIANT / REVIEW / NON-COMPLIANT` per image with concrete violations:

* identity below the drift boundary, or below the required similarity → `REVIEW`/`NON-COMPLIANT`;
* a locked group below its minimum → violation;
* no face detected → `NON-COMPLIANT`.

**Report** (`report.py`, ReportLab, A4): masthead + summary counts, per-image summary table with
thumbnails and verdicts, then per-image blocks with the group-score table, the violations, the lock
recommendations (measured values), the YouCam named-attribute changes, and the full-width
"reference → variant" measurement table (top 12 changes).

**Report ID** — `SHA-256` over the PNG bytes of the reference and every variant plus their names,
cosines and verdicts, first 16 hex characters upper-cased: change one pixel or one verdict and the ID
changes (`report.py:39`). It makes the PDF self-checkable, and it is labelled as a review artefact,
not legal proof.

---

## 12. Web UI

`streamlit run app.py` → six tabs, custom design system (`ui.py`: graphite + bronze + bone, one
metallic accent, three hairline weights, one signature "aperture" shape repeated in masthead, labels
and chips; status is carried by **glyph and density** — filled / half / crossed / hollow — never by
traffic-light hues; charts use a metallic tonal ramp instead of a categorical rainbow):

1. **▤ Campaign QA** — reference panel (faces in frame, face coverage, apparent age/sex, information
   only), one-click AI variant generation (YouCam effect or local simulation), upload of variants
   from any generator, verdict table, per-image preserved/changed/lost, group bar charts, full
   measurement breakdown, CSV download.
2. **↗ Stress Test** — mutation multiselect, run, two curves (identity / worst attribute), breakpoint
   table, the 11-step visual strip per mutation with status and worst group.
3. **◉ Identity Lock** — effect picker (aging / hair / makeup / smile / skin), aging direction
   (Older / Younger / Specific age), hair template browser, makeup reference (URL or local file),
   max SAFE / max REVIEW intensity metrics, original vs full effect vs locked, the aging series with
   per-age identity, the identity curve.
4. **▭ Campaign Report** — talent/campaign, minimum similarity, locked groups, per-image verdicts,
   PDF build + download (`faceprint_report.pdf`).
5. **◈ YouCam attributes** — live named-attribute comparison with strictness selector, two CSV
   downloads (scores by group, list of changes).
6. **◎ Calibration** — current thresholds with their source and the genuine/impostor histograms.

Upload fields accept **JPG, JPEG, JPE, JFIF, PJPEG, PJP and PNG** — the decoder reads the *content*,
not the extension (`.jpe`/`.jfif` are JPEG under other names). A "Use demo face" button boots the
bundled sample so the app is explorable without any asset.

---

## 13. CLI reference

```bash
python -m faceprint qa original.jpg ai_*.jpg --out out        # verdict + group scores -> out/qa_report.csv
python -m faceprint stress samples/demo_face.jpg --out out    # 110 mutated steps -> out/stress.csv + images
python -m faceprint calibrate data/lfw --max-people 400       # writes faceprint/calibration.json
python -m faceprint youcam original.jpg ai_1.jpg https://…    # named-attribute diff (local paths or public URLs)
python -m faceprint lock photo.jpg --effect aging --age 25    # curve CSV + *_full.jpg + *_safe.jpg
python -m faceprint lock photo.jpg --effect hair --template female_blunt_bob
python -m faceprint lock photo.jpg --effect makeup --ref look.jpg
python -m faceprint lock photo.jpg --effect smile
python -m faceprint lock photo.jpg --effect skin
python -m faceprint campaign original.jpg ai_*.jpg --policy samples/policy.json --talent "Name" --campaign "SS26" --out out
```

---

## 14. Technology stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.10+ (`X \| None` unions, walrus, dataclasses) | expressiveness for the measurement code |
| Face model | **InsightFace `buffalo_l`** (SCRFD detection + ArcFace recognition + 2D-106 landmarks + genderage) via `insightface>=0.7.3`, `onnxruntime` CPU provider, `ctx_id=-1`, `det_size=(640,640)` | one model pack, CPU-only, well-known, ~280 MB download on first run (cached) |
| Image maths | `opencv-python-headless>=4.9` (warp, NCC, Laplacian, bilateral, Lab, erosion, remap) | precise low-level control the measurements need |
| Numerics / tables | `numpy>=1.26`, `pandas>=2.0` | measurement vectors, CSV reports, chart frames |
| Web app | `streamlit>=1.35` (custom CSS design system, `st.vega_lite_chart` for charts) | fast, self-contained demo surface; no JS build step |
| PDF | `reportlab>=4.0` (A4, platypus tables, TTF Unicode font registration: DejaVu / Arial Unicode / Windows Arial) | client-facing artefact with Unicode names |
| External API | YouCam (Perfect Corp) S2S v2.0/v2.1 REST, `urllib` only (no SDK), Bearer auth from env | zero extra dependencies, auditable requests |
| Calibration data | LFW (Labeled Faces in the Wild), `<person>/*.jpg` layout | standard, public, labelled |
| Charts | Vega-Lite specs emitted by `ui.py` (metallic ramp, no built-in legend) | total visual control, no chart library |
| Secrets | `YOUCAM_API_KEY` env var only (`.env.example` documents it) | never in code, never committed |
| Data formats in | JPEG/PNG bytes (decoded by content, not extension) | tolerant uploaders |
| Data formats out | CSV (`qa_report.csv`, `faceprint_qa.csv`, `*_curve.csv`, `stress.csv`, `youcam_*.csv`), PDF, JSON (`calibration.json`, `policy.json`, `evidence.json`, raw API responses), JPEG images | drops straight into existing review workflows |

There is **no database, no server-side state**: Streamlit session state plus files on disk. The only
persistent artefacts are the calibration thresholds and the policies.

---

## 15. Configuration and secrets

```bash
pip install -r requirements.txt
export YOUCAM_API_KEY="…"        # never commit it; rotate if it was ever shown publicly
# optional: export YOUCAM_STRICTNESS=low   # default for in-app generation
```

* `faceprint/calibration.json` — thresholds + histograms (regenerate with `faceprint calibrate`).
* `samples/policy.json`, `demo_data/policy.json` — talent policy examples.
* `.streamlit/config.toml` — dark theme, bronze primary, usage stats disabled.

---

## 16. Tests

```bash
python tests/smoke.py            # end-to-end sanity; downloads ~280 MB of models on first run
python tests/test_attributes.py  # robustness to benign edits + sensitivity to the intended edit
python tests/test_lips.py        # lip-colour regression: erosion follows lip size; interior excluded
python tests/test_lock.py        # identity-lock curve + verified YouCam request shapes (mocked API)
python tests/test_multiface.py   # two faces in frame -> the matching one is compared
python tests/test_policy_edge.py # no-face image through QA, policy and PDF without crashing
python tests/test_report.py      # verdicts + PDF build
python tests/test_ui.py          # chart specs stay Vega-valid; not-measurable never scored as black
python tests/test_youcam.py      # offline attribute comparison + mocked File API / task / poll flow
```

What is covered: benign invariance (JPEG-40 re-encode, 0.6× resize → every group ≥ 85), targeted
sensitivity (each mutation must drop *its* category below a stated limit), measurement honesty
(`None` for unmeasurable, never fake black), multi-face correctness, edge cases, request-body
shapes for all five YouCam effects, and the lock solver's contract (`curve[0]` SAFE, last step not
SAFE, `safe ≤ review`). The YouCam network layer is mocked so the suite runs offline; the live path
is proven by `scripts/live_check.py`.

---

## 17. Evidence and proof

### 17.1 Live proof against the real YouCam API — `evidence/20261010_152254/`

Produced by `python scripts/live_check.py demo_data/original.jpg --effects aging hair smile`
(2026-10-10 15:22–15:24). **12/12 steps OK** (`evidence.json`, `all_ok: true`):

| Step | Seconds |
|---|---|
| `local_arcface_baseline` | 8.0 |
| `youcam_upload_and_attributes` | 13.0 |
| `youcam_generate_aging` → `youcam_attributes_aging` → `identity_lock_aging` | 14.8 → 14.5 → 9.0 |
| `youcam_generate_hair` → `youcam_attributes_hair` → `identity_lock_hair` | 14.3 → 9.7 → 5.1 |
| `youcam_generate_smile` → `youcam_attributes_smile` → `identity_lock_smile` | 12.7 → 12.1 → 8.1 |
| `pdf_report` | 0.8 |

Contents of the folder:

* `evidence.json` — per-step status/timing, effect summaries, the calibrated thresholds in force,
  the SHA-256 of the input photo, the resulting `report_id` (`41E8B84475F66A5A`);
* `raw_attributes_original.json`, `raw_generate_aging.json`, `raw_generate_hair.json`,
  `raw_generate_smile.json` — the **real API response shapes** (aging returns the whole `res_age`
  12→70 series in one call; hair/smile return a single presigned URL);
* `aging_curve.csv`, `hair_curve.csv`, `smile_curve.csv` — 11-point identity curves
  (aging: 1.00 → 0.57, SAFE until 0.7; hair: 1.00 → 0.868, SAFE to 1.0; smile: 1.00 → 0.80);
* `aging_safe.jpg`, `hair_safe.jpg`, `smile_safe.jpg`, `*_full.jpg` — full-effect and locked images;
* `report.pdf` — the generated campaign report;
* `aging_full.jpg`, `hair_full.jpg`, `smile_full.jpg` — the full-effect generations.

YouCam named-attribute changes captured in that run (examples): aging `faceshape Square → Oval`,
`cheekbones Flat → Round Cheeks`, `eye setting Average → Wide-set`, `brow left shape Straight →
Rounded`, `age 22 → 46`; hair `faceshape Square → Round`, `face aspect ratio 1.48 → 1.23`,
`horizontal third [34,34,32] → [27,37,36]`, `brow gap Far-Apart → Average`, `nose width Average →
Narrow`.

### 17.2 Threshold calibration

`faceprint/calibration.json` — 200 LFW identities, **2 923 genuine / 5 000 impostor** pairs; safe
**0.6723** (genuine median), review **0.1468** (impostor 99th percentile, FAR ≈ 1 %); both
histograms are stored bin-by-bin and rendered in the Calibration tab.

### 17.3 Offline demo set — `demo_data/`

Built by `scripts/make_demo_data.py` from `demo_data/original.jpg` (a clean frontal studio portrait,
48 % frame coverage — see `CREDITS.json` for its restricted licence and the instruction to replace
it with an owned/licensed photo before any public use). The 7 campaign variants are **simulated**
(local OpenCV mutations) and labelled as such; the results show the tool's range:

| Image | Similarity | Status | Policy verdict | Issues |
|---|---|---|---|---|
| `01_luxury_grade.jpg` | 0.968 | ● SAFE | COMPLIANT | — |
| `02_soft_glam.jpg` | 0.930 | ● SAFE | COMPLIANT | — |
| `03_eyes_pop.jpg` | 0.527 | ◐ REVIEW | NON-COMPLIANT | similarity below required 0.67; locked `Eyes` 32 < 70 |
| `04_editorial_slim.jpg` | 0.871 | ● SAFE | NON-COMPLIANT | locked `Face shape` 41 < 70 |
| `05_beauty_max.jpg` | 0.539 | ◐ REVIEW | NON-COMPLIANT | similarity below 0.67; `Face shape` 54 < 70 |
| `06_cyberpunk.jpg` | 0.556 | ◐ REVIEW | NON-COMPLIANT | similarity below 0.67; `Face shape` 69 < 70 |
| `07_smile_lipstick.jpg` | 0.848 | ● SAFE | COMPLIANT | — |

with `policy.json`, `campaign_results.csv`, `stress_test.csv` (all 10 mutations × 11 strengths) and
`faceprint_report.pdf`. Note row 4: **SAFE by identity yet NON-COMPLIANT by policy** — the case a
verdict-only tool cannot see.

### 17.4 Human-agreement study — `scripts/human_study.py`

`make` builds a blind HTML questionnaire (original vs variant, 1 = definitely different … 5 =
definitely the same) with the score key kept private; `analyze` merges raters' CSVs and reports the
mean/std per pair and the **Spearman correlation between human ratings and FACEPRINT cosine**. Ship
this number alongside the tool instead of claiming accuracy.

---

## 18. Demo playbook

`DEMO.md` holds the full script (proof checklist, 3-minute flow, say/don't-say list). The short
version:

1. **Problem (20 s)** — "A brand pays for a face, then generates 30 AI visuals. Which still look like
   the person they hired?"
2. **Campaign QA (60 s)** — original + variants → SAFE / REVIEW / DRIFT table; open one REVIEW image:
   preserved vs lost groups + the YouCam change (`faceshape: Round → Oval`).
3. **The surprise (30 s)** — an image that *looks* fine but scores badly (or vice versa) — proof the
   tool tells you something the eye doesn't.
4. **Identity Lock (40 s)** — "aging is SAFE up to 70 % intensity for this model" → locked vs full.
5. **Report (20 s)** — PDF with Report ID: "this is what goes to the client."
6. **Honesty (10 s)** — "a review aid, not identity proof; thresholds calibrated on LFW; n raters
   agree at ρ = …".

Say: *visual similarity*, *identity drift flagged for human review*, *independent QA layer for AI
advertising*. Don't say: *"99 % same person"*, *legal proof*, *biometric identification*,
*guarantees likeness rights*. And say plainly that Identity Lock is a **blend-back dial** for the
effects without a native strength parameter — not a hidden strength control of the YouCam effect.

---

## 19. Honest limitations

* The verdict is **cosine similarity of ArcFace embeddings** — not a probability of identity, not a
  biometric identification result, not legal evidence.
* The **attribute breakdown is a heuristic**: identity thresholds are calibrated on LFW, but the
  group weights and tolerances were chosen empirically against the local mutations
  (`tests/test_attributes.py`). Very fine changes (below ~10 %) can be missed.
* **Expression and pose** are 2-D landmark proxies (smile, mouth opening, nose offsets), not a full
  3-D pose. **Age/sex** come from InsightFace `genderage` (~±5 years) and are reported, not scored.
* **Hair and makeup** are not segmented locally (only lip colour is measured): a hairstyle change is
  visible through ArcFace and through the YouCam named attributes, not through the 8 local groups.
* **Several faces in the frame**: the face most similar to the reference is compared (and the advice
  row says so); the reference photo itself should contain one person — the UI warns when it doesn't.
* The **local mutations are coarse 2-D approximations**; real aging/hair/makeup/smile/skin are done
  through YouCam (Identity Lock).
* `mu-transfer` and `skin-simulation` reach the engine but depend on YouCam's content rules
  (`error_inappropriate_ref_case01`, `error_large_face_angle` on non-frontal frames).
* Face Analyzer calls are **billed** (30 units per call) — hence the separate tab, never implicit.
* Use only with the model's consent. The bundled demo face is a placeholder with a restricted
  licence (`demo_data/CREDITS.json`) and must be replaced before public use.

---

## 20. Integrating another generator

The product is generator-agnostic on the QA side: upload **any** generator's output (Firefly,
Midjourney, in-house diffusion) in Campaign QA — no integration needed.

For the automated variant, write a function that calls the API and returns a **BGR image**, then
register it next to the YouCam paths in `faceprint/generators.py` (the YouCam client is already there
as the reference implementation of the upload → task → poll → decode pattern). The stress test uses
the local OpenCV mutations as the offline stand-in.

---

## 21. Project structure

```
faceprint_project/
├── app.py                          Streamlit entry: 6 tabs, uploaders, wiring
├── DEMO.md                         proof checklist, 3-minute demo script, say/don't-say
├── README.md                       this file
├── requirements.txt                insightface, onnxruntime, opencv, numpy, pandas, streamlit, reportlab
├── .env.example                    YOUCAM_API_KEY template
├── .streamlit/config.toml          dark theme, bronze primary, usage stats off
├── faceprint/
│   ├── engine.py                   InsightFace buffalo_l wrapper; Face dataclass; cosine
│   ├── qa.py                       thresholds, status mapping, Baseline (Identity Lock), check(), stress_sweep()
│   ├── attributes.py               measure / compare / explain; canonical frame; tolerances; NAMES
│   ├── transforms.py               10 local mutations (strength 0-1)
│   ├── calibrate.py                genuine vs impostor calibration -> calibration.json
│   ├── youcam.py                   YouCam S2S client (File API, task, poll, templates) + attribute diff
│   ├── generators.py               effect -> endpoint table, framing, compositing, generate()
│   ├── lock.py                     blend sweep + native strength sweep (max SAFE intensity)
│   ├── policy.py                   talent policy + COMPLIANT/REVIEW/NON-COMPLIANT verdict
│   ├── report.py                   PDF campaign report + SHA-256 Report ID
│   ├── ui.py                       design system (CSS, chips, Vega-Lite charts, legends)
│   ├── cli.py                      qa / stress / calibrate / youcam / lock / campaign
│   ├── calibration.json            calibrated thresholds + histograms (200 identities, LFW)
│   ├── __init__.py / __main__.py   package marker; python -m faceprint
├── samples/
│   ├── demo_face.jpg               bundled demo reference (placeholder, replace)
│   ├── policy.json                 sample talent policy
│   └── youcam_sample.json          captured real response for the offline comparison test
├── demo_data/                      OFFLINE demo set (simulated campaign, labelled as such)
│   ├── original.jpg                reference model photo (licence: RESTRICTED - see CREDITS.json)
│   ├── campaign/01…07_*.jpg        7 simulated variants
│   ├── policy.json                 {"Face shape": 70, "Eyes": 70}
│   ├── campaign_results.csv        similarity / status / policy verdict / issues
│   ├── stress_test.csv             10 mutations x 11 strengths + breakpoints
│   ├── faceprint_report.pdf        generated client report
│   └── CREDITS.json                provenance + licence + replace-before-public warning
├── evidence/
│   └── 20261010_152254/            LIVE proof: 12/12 steps OK
│       ├── evidence.json           step timings, effect summaries, report_id
│       ├── raw_*.json              real YouCam API responses (attributes, aging, hair, smile)
│       ├── aging/hair/smile_curve.csv    identity curves (11 points each)
│       ├── *_safe.jpg / *_full.jpg       locked and full-effect images
│       └── report.pdf              the report generated by the live run
├── scripts/
│   ├── live_check.py               end-to-end live proof against YouCam -> evidence/<ts>/
│   ├── get_lfw.py                  download + arrange LFW for calibration
│   ├── human_study.py              blind rating study -> Spearman(human, cosine)
│   └── make_demo_data.py           rebuild demo_data/ from any photo
└── tests/
    ├── smoke.py                    end-to-end sanity (first run downloads ~280 MB models)
    ├── test_attributes.py          benign invariance + targeted sensitivity
    ├── test_lips.py                lip-colour measurement regressions
    ├── test_lock.py                lock solver + verified YouCam request shapes (mocked)
    ├── test_multiface.py           multi-face matching rule
    ├── test_policy_edge.py         no-face edge cases through QA/policy/report
    ├── test_report.py              verdicts + PDF build
    ├── test_ui.py                  chart specs + not-measurable rule
    └── test_youcam.py              offline attribute comparison + mocked API flow
```

---

## 22. Ethics, consent and legal position

* FACEPRINT is a **visual-similarity review aid**. It is not a biometric identification system, not an
  authenticity proof, and not legal evidence. It exists to *flag faces for human review*.
* Use it only on photos of **consenting** people. The demo asset is a placeholder with a restricted
  licence and is marked for replacement precisely because the project is about consent and image
  rights.
* The YouCam API key lives only in `YOUCAM_API_KEY` — never in code, examples or commits. Rotate any
  key that was ever displayed publicly.

---

## 23. Quick start

```bash
git clone <repo> && cd faceprint_project
pip install -r requirements.txt

python tests/smoke.py                       # first run downloads ~280 MB of InsightFace models
streamlit run app.py                        # the 6-tab UI
python -m faceprint stress samples/demo_face.jpg --out out
python -m faceprint qa demo_data/original.jpg demo_data/campaign/*.jpg --out out
python -m faceprint campaign demo_data/original.jpg demo_data/campaign/*.jpg \
       --policy demo_data/policy.json --talent "Demo model" --campaign "SS26 (simulated)" --out out

export YOUCAM_API_KEY=…                     # optional, for the live layers
python -m faceprint youcam demo_data/original.jpg demo_data/campaign/01_luxury_grade.jpg
python -m faceprint lock demo_data/original.jpg --effect aging --age 25 --out out
python scripts/live_check.py demo_data/original.jpg --effects aging hair smile   # writes evidence/<ts>/
python scripts/get_lfw.py && python -m faceprint calibrate data/lfw             # recalibrate thresholds
python scripts/make_demo_data.py your_photo.jpg                                 # rebuild the offline demo set
```
