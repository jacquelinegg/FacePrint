"""Interpretable facial measurements.
Two independent signals per category, averaged 50/50:
   * geometry   - ratios from the 106-point landmarks (scale/rotation invariant, normalised by inter-ocular distance);
   * appearance - the face is warped to a canonical frame (eyes + mouth centre) and each region is compared with a
                  local-contrast-normalised cross-correlation (NCC). Unlike landmarks this sees what the pixels
                  actually look like, so enlarged eyes or a slimmed jaw that the landmark model "follows" are still caught.
Extra groups: Expression (smile / mouth open), Head pose (yaw / pitch proxy).
Apparent age / predicted sex are reported as information in `explain()` but not scored: the age model's own error
(~5 years, dead zone already applied) means the score is pinned at 100 and carries no signal."""
import cv2, numpy as np

# 2d106 layout (verified visually): contour 0-32, eyes 33-42 / 87-96,
# brows 43-51 / 97-105, mouth 52-71, nose 72-86
SPEC = {  # name: (category, mode, tolerance)  tolerance = "clearly noticeable change"
    "face_ratio":  ("Face shape", "rel", 0.08), "jaw_width":  ("Face shape", "rel", 0.06),
    "eye_size":    ("Eyes", "rel", 0.08),       "eye_open":   ("Eyes", "rel", 0.15),
    "eye_spacing": ("Eyes", "rel", 0.08),
    "brow_len":    ("Brows", "rel", 0.15),      "brow_gap":   ("Brows", "rel", 0.20),
    "brow_slope":  ("Brows", "abs", 6.0),
    "lip_width":   ("Lips", "rel", 0.12),       "lip_full":   ("Lips", "rel", 0.25),
    "nose_width":  ("Nose", "rel", 0.12),       "nose_len":   ("Nose", "rel", 0.12),
}
CATEGORIES = ["Face shape", "Eyes", "Brows", "Lips", "Nose", "Skin tone", "Expression", "Head pose"]
SKIN_TOL = 10.0  # CIELAB delta-E
SKIN_TEX_TOL = 0.45  # relative change of the high-frequency skin energy
LIP_TOL = 8.0    # delta-E of lip colour (catches lipstick / colour grading)
# extra geometric signals: name -> (category, tolerance in absolute units)
# tolerances sit ~5x above the measured noise floor (JPEG / resize / 6 deg rotation): smile .003, yaw .004, pitch .004
EXTRA = {"smile": ("Expression", 0.015), "mouth_open": ("Expression", 0.04),
         "yaw": ("Head pose", 0.025), "pitch": ("Head pose", 0.03)}
AGE_TOL = 7.0    # years beyond a +-4 dead zone (InsightFace age error is itself ~+-5)
APPEAR_LO = 0.5  # NCC at/below which a region counts as completely different (score 0); 1.0 -> 100

# canonical frame (256x256): eye centres + mouth centre are pinned here
CAN = 256
DST = np.float32([[88, 100], [168, 100], [128, 184]])
REGIONS = {"Eyes": [(88, 100, 56, 36), (168, 100, 56, 36)], "Brows": [(88, 76, 64, 28), (168, 76, 64, 28)],
           "Nose": [(128, 142, 48, 56)], "Lips": [(128, 184, 76, 44)]}
JAW_Y = 150      # Face shape: contour edges of the lower face, y >= JAW_Y

def _ptp(a, axis): return float(np.ptp(a[:, axis]))

def _aligned(lm):
    a, b = lm[33:43].mean(0), lm[87:97].mean(0)
    if a[0] > b[0]: a, b = b, a
    ang = np.arctan2(b[1] - a[1], b[0] - a[0])
    c, s = np.cos(-ang), np.sin(-ang)
    R = np.array([[c, -s], [s, c]])
    return (lm - a) @ R.T, float(np.linalg.norm(b - a))

def _skin_lab(img, lm, iod):
    ec = [lm[33:43].mean(0), lm[87:97].mean(0)]
    mc, mid = lm[52:72].mean(0), (ec[0] + ec[1]) / 2
    labs = []
    for e in ec:
        p = 0.45 * e + 0.55 * mc
        p = p + (e - mid) / (np.linalg.norm(e - mid) + 1e-6) * 0.15 * iod
        r = max(2, int(0.08 * iod)); x, y = int(p[0]), int(p[1])
        patch = img[max(0, y - r):y + r, max(0, x - r):x + r]
        if patch.size:
            lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
            labs.append(np.median(lab, 0) * [100 / 255, 1, 1] - [0, 128, 128])
    return np.mean(labs, 0) if labs else None      # None = not measurable, NOT black

def _skin_texture(img, lm, iod=None):
    """High-frequency energy (pores / fine lines) in the canonical frame: forehead + both cheeks.

    Aging and every beauty filter move this signal far more than they move the skin *colour* - without
    it a heavily smoothed or aged face scores as 'unchanged'. The canonical frame is what keeps it
    honest: the patches sit at fixed anatomical positions, so an eye enlargement, a lip fill, a smile
    or a jaw warp must not register as a skin change, and a pure colour grade must not either. The
    three patches are averaged, so one region's resampling noise does not drive the score."""
    can = canonical(img, lm)
    g = cv2.cvtColor(can, cv2.COLOR_BGR2GRAY)
    r = max(6, int(CAN * 0.055))
    pts = [(CAN // 2, int(CAN * 0.22)), (int(CAN * 0.235), int(CAN * 0.55)), (int(CAN * 0.765), int(CAN * 0.55))]
    vals = []
    for cx, cy in pts:
        patch = g[max(0, cy - r):cy + r, cx - r:cx + r]
        if patch.size and patch.shape[0] == 2 * r and patch.shape[1] == 2 * r:
            patch = cv2.resize(patch, (48, 48), interpolation=cv2.INTER_AREA)   # resolution independent
            vals.append(float(cv2.Laplacian(patch, cv2.CV_64F).var()))
    return float(np.mean(vals)) if vals else None

def canonical(img, lm):
    """Warp the face into the canonical frame (similarity-like affine from eyes + mouth centre)."""
    a, b = lm[33:43].mean(0), lm[87:97].mean(0)
    if a[0] > b[0]: a, b = b, a
    M = cv2.getAffineTransform(np.float32([a, b, lm[52:72].mean(0)]), DST)
    return cv2.warpAffine(img, M, (CAN, CAN), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def _texture(c):
    g = cv2.GaussianBlur(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
    d = g - cv2.GaussianBlur(g, (0, 0), 12)
    return d / (np.sqrt(cv2.GaussianBlur(d * d, (0, 0), 12)) + 4)       # local contrast normalisation

def _edges(c):
    g = cv2.GaussianBlur(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 2)
    return cv2.GaussianBlur(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), 3)

def _crop(x, box):
    cx, cy, w, h = box
    return x[cy - h // 2:cy + h // 2, cx - w // 2:cx + w // 2]

def _ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))

def _lip_lab(img, lm):
    """Median CIELAB of the lip ring.

    The erosion is relative to the LIP size, not to the image: YouCam returns 512-1255px images while
    the lips are ~50x20px, so an image-proportional erosion erased the whole mask and the lip colour
    silently became "not measured" - or, when a few pixels survived, was compared against an even
    tighter mask in the reference and scored as fake drift.
    The measurement is the RING between two erosions: lip tissue, never the mouth interior
    (teeth / tongue / shadow) that an open mouth puts in the middle of the hull.
    Unmeasurable (mask under 40 px, near-black median, lips out of frame) returns None, so the caller
    skips it instead of scoring a fake black."""
    mouth = lm[52:72].astype(np.int32)
    if not np.isfinite(mouth).all(): return None
    x, y, w, h = cv2.boundingRect(mouth)
    if w <= 0 or h <= 0 or w < 8 or h < 4: return None
    outer = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(outer, cv2.convexHull(mouth), 255)
    e1 = max(1, int(round(0.10 * h)))                      # inside the lip line
    e2 = max(2, int(round(0.28 * h)))                      # ring = e1..e2 from the lip line
    ring = cv2.subtract(cv2.erode(outer, np.ones((2 * e1 + 1, 2 * e1 + 1), np.uint8)),
                        cv2.erode(outer, np.ones((2 * e2 + 1, 2 * e2 + 1), np.uint8)))
    sel = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[ring > 0].astype(float)
    if len(sel) < 40: return None                          # eroded away (or lips out of frame)
    med = np.median(sel, 0) * [100 / 255, 1, 1] - [0, 128, 128]
    if med[0] < 10: return None                            # near-black: interior/hair, not lips
    return med

def measure(img, lm, face=None):
    P, iod = _aligned(lm)
    cont = P[0:33]; w, h = _ptp(cont, 0), _ptp(cont, 1)
    low = cont[cont[:, 1] > cont[:, 1].min() + 0.7 * h]
    eyes = [P[33:43], P[87:97]]; brows = [P[43:52], P[97:106]]
    ang = [np.degrees(np.arctan(np.polyfit(b[:, 0], b[:, 1], 1)[0])) for b in brows]
    m, n = P[52:72], P[72:87]
    corners = m[[m[:, 0].argmin(), m[:, 0].argmax()]]
    eye_y, mouth_y = np.mean([e.mean(0)[1] for e in eyes]), m.mean(0)[1]
    can = canonical(img, lm)
    out = {
        "face_ratio": w / h, "jaw_width": _ptp(low, 0) / w,
        "eye_size": np.mean([_ptp(e, 0) for e in eyes]) / iod,
        "eye_open": np.mean([_ptp(e, 1) / _ptp(e, 0) for e in eyes]),
        "eye_spacing": iod / w,
        "brow_len": np.mean([_ptp(b, 0) for b in brows]) / iod,
        "brow_gap": np.mean([e.mean(0)[1] - b.mean(0)[1] for e, b in zip(eyes, brows)]) / iod,
        "brow_slope": (ang[1] - ang[0]) / 2,
        "lip_width": _ptp(m, 0) / iod, "lip_full": _ptp(m, 1) / _ptp(m, 0),
        "nose_width": _ptp(n, 0) / iod, "nose_len": _ptp(n, 1) / iod,
        "skin_lab": _skin_lab(img, lm, iod), "lip_lab": _lip_lab(img, lm),
 "skin_texture": _skin_texture(img, lm, iod),
        # expression: corners above the mouth centre = smile (+), mouth opening relative to eye distance
        "smile": float((m[:, 1].mean() - corners[:, 1].mean()) / iod), "mouth_open": _ptp(m, 1) / iod,
        # pose proxies (2-D, no 3-D model): nose offset from the face midline, nose height between eyes and mouth
        "yaw": float((n[:, 0].mean() - (cont[:, 0].min() + cont[:, 0].max()) / 2) / w),
        "pitch": float((n[:, 1].mean() - eye_y) / (mouth_y - eye_y + 1e-6)),
        "texture": _texture(can), "edges": _edges(can),
        "age": getattr(face, "age", None), "gender": getattr(face, "gender", None),
    }
    return out

def appearance(m0, m1):
    """Per-category NCC (-1..1) of canonical-frame regions, plus the lower-face contour."""
    r = {c: float(np.mean([_ncc(_crop(m0["texture"], b), _crop(m1["texture"], b)) for b in bs])) for c, bs in REGIONS.items()}
    r["Face shape"] = _ncc(m0["edges"][JAW_Y:], m1["edges"][JAW_Y:])
    return r

def _app_score(ncc):
    return 100 * float(np.clip((ncc - APPEAR_LO) / (1 - APPEAR_LO), 0, 1))

def compare(m0, m1):
    """Return ({category: 0-100 score}, {attribute: 0-100 score})."""
    detail = {}
    for k, (_, mode, tol) in SPEC.items():
        d = abs(m1[k] - m0[k]) / (abs(m0[k]) + 1e-6) if mode == "rel" else abs(m1[k] - m0[k])
        detail[k] = 100 * float(np.exp(-0.5 * d / tol))
    de = float(np.linalg.norm(np.asarray(m1["skin_lab"]) - np.asarray(m0["skin_lab"]))) \
        if m0.get("skin_lab") is not None and m1.get("skin_lab") is not None else None
    if de is not None: detail["skin_tone"] = 100 * float(np.exp(-0.5 * de / SKIN_TOL))
    dl = float(np.linalg.norm(np.asarray(m1["lip_lab"]) - np.asarray(m0["lip_lab"]))) \
        if m0.get("lip_lab") is not None and m1.get("lip_lab") is not None else None
    if dl is not None: detail["lip_color"] = 100 * float(np.exp(-0.5 * dl / LIP_TOL))
    if m0.get("skin_texture") is not None and m1.get("skin_texture") is not None:
        rel = abs(m1["skin_texture"] - m0["skin_texture"]) / (abs(m0["skin_texture"]) + 1e-6)
        detail["skin_texture"] = 100 * float(np.exp(-0.5 * rel / SKIN_TEX_TOL))
    for k, (_, tol) in EXTRA.items():
        detail[k] = 100 * float(np.exp(-0.5 * abs(m1[k] - m0[k]) / tol))
    app = appearance(m0, m1)
    for c, v in app.items(): detail[f"appearance:{c}"] = _app_score(v)
    cats = {}
    for c in CATEGORIES:
        geo = [detail[k] for k, v in SPEC.items() if v[0] == c]
        geo += [detail[k] for k, v in EXTRA.items() if v[0] == c]
        if c == "Lips" and "lip_color" in detail: geo.append(detail["lip_color"])
        if c == "Skin tone":
            geo = [detail[k] for k in ("skin_tone", "skin_texture") if k in detail] or geo
            if len(geo) > 1:            # drop if EITHER colour or texture moved - averaging hides one of them
                geo = [min(geo)]
        if not geo: continue                      # nothing measurable for this group in this pair
        score = float(np.mean(geo))
        key = "Lips" if c == "Expression" else c          # expression is also judged on how the mouth region looks
        if key in app: score = 0.5 * score + 0.5 * detail[f"appearance:{key}"]
        cats[c] = score
    return cats, detail

# ---- human-readable diff ------------------------------------------------------------------------
# detail key -> (display name, unit kind, key in the measurement dict)
NAMES = {
    "face_ratio":  ("Face ratio (width / height)", "ratio", "face_ratio"),
    "jaw_width":   ("Jaw width / face width", "ratio", "jaw_width"),
    "eye_size":    ("Eye width / inter-ocular distance", "ratio", "eye_size"),
    "eye_open":    ("Eye openness (height / width)", "ratio", "eye_open"),
    "eye_spacing": ("Eye spacing / face width", "ratio", "eye_spacing"),
    "brow_len":    ("Brow length / inter-ocular distance", "ratio", "brow_len"),
    "brow_gap":    ("Brow-to-eye gap / inter-ocular distance", "ratio", "brow_gap"),
    "brow_slope":  ("Brow slope difference (deg)", "deg", "brow_slope"),
    "lip_width":   ("Lip width / inter-ocular distance", "ratio", "lip_width"),
    "lip_full":    ("Lip fullness (height / width)", "ratio", "lip_full"),
    "nose_width":  ("Nose width / inter-ocular distance", "ratio", "nose_width"),
    "nose_len":    ("Nose length / inter-ocular distance", "ratio", "nose_len"),
    "smile":       ("Smile (mouth-corner lift)", "signed", "smile"),
    "mouth_open":  ("Mouth opening / inter-ocular distance", "signed", "mouth_open"),
    "yaw":         ("Head yaw (turn)", "signed", "yaw"),
    "pitch":       ("Head pitch (tilt)", "signed", "pitch"),
    "skin_tone":   ("Skin tone colour", "color", "skin_lab"),
    "skin_texture": ("Skin texture (fine lines)", "pct", "skin_texture"),
    "lip_color":   ("Lip colour", "color", "lip_lab"),
    "age":         ("Apparent age (model estimate)", "age", "age"),
    "gender":      ("Predicted sex (model estimate)", "gender", "gender"),
}
for _c in REGIONS:                                   # appearance NCC rows, one per compared region
    NAMES[f"appearance:{_c}"] = (f"{_c} region appearance", "ncc", None)

def _colour_words(d, what="colour"):
    """Plain-language direction of a CIELAB difference (d = variant - reference), without the numbers -
    the numbers stay in the reference/variant columns for whoever wants them, the note is for a human.
    `what` is 'lip' or 'skin' so the phrasing reads naturally in each row."""
    w = []
    if d[0] < -2: w.append("darker")
    elif d[0] > 2: w.append("lighter")
    if d[1] > 2: w.append("redder" if what == "lip" else "more red / saturated")
    elif d[1] < -2: w.append("greener")
    if d[2] > 2: w.append("warmer")
    elif d[2] < -2: w.append("cooler")
    if not w: return "essentially unchanged"
    if what == "lip": return "the lipstick became " + " and ".join(w)
    return "the skin became " + " and ".join(w)

def _category_of(src):
    """Which category a measured value belongs to (information-only values -> 'Information')."""
    if src in SPEC: return SPEC[src][0]
    if src in EXTRA: return EXTRA[src][0]
    return {"skin_lab": "Skin tone", "lip_lab": "Lips"}.get(src, "Information")

def explain(m0, m1, detail):
    """One row per measured attribute: reference value, variant value, magnitude, 0-100 score and a
    plain-language note, most-changed first. The 'what exactly did the AI change' table behind the scores."""
    rows = []
    for key, (name, kind, src) in NAMES.items():
        if kind == "ncc":
            nv = detail.get(key)
            if nv is None: continue
            rows.append({"category": key.split(":", 1)[1], "attribute": name, "reference": "1.00",
                         "variant": f"{nv / 100:.2f}", "change": f"NCC {nv / 100:.2f}", "score": round(nv),
                         "note": "region pixel match; 1.00 = identical"})
            continue
        v0, v1 = m0.get(src), m1.get(src)
        if v0 is None or v1 is None: continue          # not measurable on this side -> no guess
        cat = _category_of(src)
        if kind == "color":
            d = np.asarray(v1, float) - np.asarray(v0, float); de = float(np.linalg.norm(d))
            what = "lip" if key == "lip_color" else "skin"
            rows.append({"category": cat, "attribute": name,
                         "reference": f"L*{v0[0]:.0f} a*{v0[1]:+.0f} b*{v0[2]:+.0f}",
                         "variant": f"L*{v1[0]:.0f} a*{v1[1]:+.0f} b*{v1[2]:+.0f}", "change": f"dE {de:.1f}",
                         "score": round(detail[key]), "note": _colour_words(d, what)})
        elif kind == "age":
            d = float(v1) - float(v0)
            rows.append({"category": cat, "attribute": name, "reference": f"{v0:.0f} y", "variant": f"{v1:.0f} y",
                         "change": f"{d:+.0f} y", "score": None,
                         "note": "model estimate only - reported, not scored (its own error is ~5 y)"})
        elif kind == "gender":
            same = int(v0) == int(v1)
            rows.append({"category": cat, "attribute": name, "reference": "male" if int(v0) else "female",
                         "variant": "male" if int(v1) else "female", "change": "same" if same else "CHANGED",
                         "score": None, "note": "model estimate only - reported, not scored"})
        elif kind == "pct":                                  # relative change of an energy/ratio value
            rel = (float(v1) - float(v0)) / abs(float(v0) + 1e-9)
            rows.append({"category": cat, "attribute": name, "reference": f"{float(v0):.1f}",
                         "variant": f"{float(v1):.1f}", "change": f"{rel:+.0%}", "score": round(detail[key]),
                         "note": "high-frequency skin detail: aging and beauty filters smooth it away"})
        else:
            d = float(v1) - float(v0); rel = d / abs(float(v0) + 1e-9)
            rows.append({"category": cat, "attribute": name,
                         "reference": f"{float(v0):+.3f}" if kind == "signed" else f"{float(v0):.3f}",
                         "variant": f"{float(v1):+.3f}" if kind == "signed" else f"{float(v1):.3f}",
                         "change": f"{rel:+.0%}" if kind == "ratio" else f"{d:+.3f}", "score": round(detail[key]),
                         "note": ""})
    rows.sort(key=lambda r: r["score"] if r["score"] is not None else 101)
    return rows
