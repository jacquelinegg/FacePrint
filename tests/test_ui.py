"""Visual-system helpers + the not-measurable rule: specs stay Vega-valid, the palette stays
hue-free, and a region with no pixels never becomes a fake "black" measurement."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import pandas as pd
import streamlit as st
import faceprint.ui as ui
from faceprint import attributes as A
import numpy as np

captured = {}
st.vega_lite_chart = lambda spec, **k: captured.clear() or captured.update(spec)
ui.st = st

# ---- vega_line: named index, unnamed index, and NaN -> null (NO FACE stays a gap) ----
df = pd.DataFrame({"strength": [0.0, 0.5, 1.0], "a": [1.0, 0.8, 0.5], "b": [1.0, 0.7, None]}).set_index("strength")
ui.vega_line(df)
assert captured["encoding"]["x"]["field"] == "strength"
json.dumps(captured, allow_nan=False)                      # NaN would be invalid JSON for Vega
assert any(v["y"] is None for v in captured["data"]["values"])
assert captured["encoding"]["color"]["legend"] is None     # no built-in legend to lose

piv = pd.DataFrame({"same person": [1, 2], "different people": [8, 3]}, index=[0.1, 0.2])
ui.vega_line(piv, (0.0, 10.0))
assert captured["encoding"]["x"]["field"] == "x" and captured["data"]["values"][0]["series"] == "same person"
ui.vega_bar([("Eyes", 47.3), ("Lips", 88.9)])
assert captured["mark"]["color"] == ui.ramp(1)[0] and len(captured["data"]["values"]) == 2

# ---- ramp + chips: monochrome glyphs, no hue-coded status ----
assert ui.ramp(6) != ui.ramp(5)[:5] and all(c.startswith("#") for c in ui.ramp(9))
assert [ui.STATUS[k] for k in ("SAFE", "REVIEW", "IDENTITY DRIFT", "NO FACE")] == ["\u25cf", "\u25d0", "\u2715", "\u25cb"]
for s in ("SAFE", "REVIEW", "IDENTITY DRIFT", "NO FACE"):
    assert f"fp-chip" in ui.chip(s) and s in ui.chip(s)
print("OK ui helpers: vega specs, ramp, chips")

# ---- not-measurable Lab: None, never zeros ----
m0 = {"skin_lab": np.array([68., 24., 34.]), "lip_lab": np.array([60., 30., 10.]), "age": 61, "gender": 0}
m1 = {"skin_lab": np.array([69., 20., 29.]), "lip_lab": None, "age": 57, "gender": 0}
for k in A.SPEC: m0[k] = m1[k] = 1.0
for k in A.EXTRA: m0[k] = m1[k] = 0.0
m0["texture"] = m1["texture"] = np.ones((A.CAN, A.CAN), np.float32)
m0["edges"] = m1["edges"] = np.ones_like(m0["texture"])
cats, detail = A.compare(m0, m1)
assert "skin_tone" in detail and "lip_color" not in detail      # measurable vs skipped
assert "Lips" in cats and "Skin tone" in cats                   # groups still score from geometry
rows = A.explain(m0, m1, detail)
assert not any("L*0 a*+0 b*+0" in str(r["variant"]) for r in rows)   # no fake black lips
assert any(r["attribute"].startswith("Apparent age") and r["score"] is None for r in rows)  # info-only row
print("OK not-measurable regions are skipped, never scored as black")
