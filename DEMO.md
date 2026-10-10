# Demo + submission playbook

## 1. Proof checklist (do in this order)
| # | Command | What it proves | Done |
|---|---|---|---|
| 1 | `export YOUCAM_API_KEY=<NEW key>` then `python scripts/live_check.py photo.jpg --effects aging hair` | Real YouCam flow: upload → attributes → generate → re-analyze → lock → PDF. Writes `evidence/<ts>/evidence.json` + raw responses | ☐ |
| 2 | `python scripts/get_lfw.py && python -m faceprint calibrate data/lfw` | Thresholds from data (same-person vs different-people histogram) | ☐ |
| 3 | `python scripts/human_study.py make ...` → send HTML to 5-10 people → `analyze` | Humans agree with the score (report the Spearman number) | ☐ |
| 4 | `python scripts/make_demo_data.py your_photo.jpg` | Offline fallback demo set (labelled *simulated*) | ☐ |

If a live step fails: send me `evidence/<ts>/error_*.txt` and `raw_*.json` - I fix the parser in minutes.

## 2. Three-minute demo
1. **Problem (20s)** "A brand pays for a face, then generates 30 AI visuals. Which still look like the person they hired?"
2. **Campaign QA (60s)** Original + 6 YouCam-generated variants → table: SAFE / REVIEW / DRIFT. Open one REVIEW image: preserved vs lost groups + YouCam attribute change (`faceshape: Round → Oval`).
3. **Surprise (30s)** Show one image that *looks* fine but the score flags (or the reverse). This is the proof the tool tells something the eye doesn't.
4. **Identity Lock (40s)** "Hair effect is SAFE up to 40% intensity for this model" → locked version next to the full effect.
5. **Report (20s)** PDF with Report ID → "this is what goes to the client". 6. **Honesty (10s)** "Review aid, not identity proof; thresholds calibrated on LFW; n raters agree ρ=…".

## 3. Say / don't say
- ✅ "visual similarity", "identity drift flagged for human review", "independent QA layer for AI advertising".
- ❌ "99% same person", "legal proof", "biometric identification", "guarantees likeness rights".
- Identity Lock = *blend-back dial*, not a native strength control of the YouCam effect. Say so.
- Only use consenting people's photos. The bundled sample face is a placeholder - replace it.

## 4. Plan to 2 Nov 2026 (check eligibility/rules on the Devpost page yourself)
- Days 1-3: live_check green (fix parsers) · Days 4-6: LFW calibration + human study · Days 7-10: real campaign with 2 effects × 2 models
- Days 11-14: polish UI (one clean flow), record demo video, 2-page write-up · Last 3 days: buffer, no new features.
