"""Identity QA: embedding similarity (verdict) + attribute breakdown (explanation)."""
import json, os
from . import attributes as A, transforms as T
from .engine import cosine

CAL_PATH = os.path.join(os.path.dirname(__file__), "calibration.json")
DEFAULT = {"safe": 0.60, "review": 0.35, "source": "uncalibrated default (run `python -m faceprint calibrate <dir>`)"}

def thresholds():
    if os.path.exists(CAL_PATH):
        return json.load(open(CAL_PATH))["thresholds"]
    return DEFAULT

def status_of(cos, th=None):
    th = th or thresholds()
    return "SAFE" if cos >= th["safe"] else "REVIEW" if cos >= th["review"] else "IDENTITY DRIFT"

ADVICE = {
    "Face shape": "keep face proportions / jaw line locked (reduce reshape strength)",
    "Eyes": "lock eye shape and spacing (eye size / openness drifted)",
    "Brows": "lock brow shape and position",
    "Lips": "lock lip width and fullness",
    "Nose": "lock nose width and length",
    "Skin tone": "constrain skin tone / colour grade - and the smoothing - to the reference",
    "Expression": "keep the original expression (smile / mouth opening changed)",
    "Head pose": "keep the original head angle (turn / tilt changed)",
}

class Baseline:
    """Identity baseline of the reference person (the 'Identity Lock')."""
    def __init__(self, engine, img):
        self.engine = engine
        self.face, self.n_ref_faces = engine.analyze(img)
        if self.face is None:
            raise ValueError("No face detected in the reference image")
        self.attrs = A.measure(img, self.face.lm, self.face)

    def check(self, img):
        """Compare the face in `img` that best matches the reference (not simply the largest one)."""
        face, n = self.engine.analyze(img, reference=self.face.embedding)
        if face is None:
            return {"status": "NO FACE", "cosine": 0.0, "categories": {}, "details": {}, "preserved": [],
                    "changed": [], "lost": [], "n_faces": 0, "advice": ["No face detected in this image"], "changes": []}
        cos = cosine(self.face.embedding, face.embedding)
        m = A.measure(img, face.lm, face)
        cats, detail = A.compare(self.attrs, m)
        pres = [c for c, v in cats.items() if v >= 80]
        chg = [c for c, v in cats.items() if 60 <= v < 80]
        lost = [c for c, v in cats.items() if v < 60]
        changes = A.explain(self.attrs, m, detail)
        advice = []
        for c in lost + chg:
            what = "; ".join(f"{r['attribute']} {r['change']}" + (f" ({r['note']})" if r["note"] and "unchanged" not in r["note"] else "")
                             for r in changes
                             if r["category"] == c and r["score"] is not None and r["score"] < 80)[:200]
            advice.append(f"{c}: {ADVICE[c]}" + (f" — measured: {what}" if what else ""))
        if n > 1:
            advice.insert(0, f"{n} faces in the frame - the one most similar to the reference was compared")
        return {"status": status_of(cos), "cosine": cos, "categories": cats, "details": detail,
                "preserved": pres, "changed": chg, "lost": lost, "n_faces": n, "advice": advice, "changes": changes}

def stress_sweep(baseline, img, steps=tuple(round(0.1 * i, 2) for i in range(11)), transforms=None):
    """Apply each mutation at growing strength; record where identity leaves SAFE / hits DRIFT and
    where the *attributes* start to drift. Two independent layers on purpose: ArcFace stays high under
    cosmetic edits (lipstick, tone) while the attribute layer shows the face really changed."""
    rows, images = [], {}
    for name in transforms or T.TRANSFORMS:
        review_at = drift_at = attr_at = None
        for s in steps:
            out = T.apply(name, img, baseline.face.lm, s)
            r = baseline.check(out)
            cats = r["categories"]
            worst = min(cats, key=cats.get) if cats else None
            wsc = cats[worst] if cats else None
            n_changed = sum(1 for v in cats.values() if v < 80)
            no_face = r["status"] == "NO FACE"
            rows.append({"transform": name, "strength": s, "cosine": None if no_face else r["cosine"],
                         "status": r["status"], "worst_attribute": worst, "worst_attribute_score": wsc,
                         "attributes_changed": n_changed})
            images[(name, s)] = out
            if not no_face and r["status"] != "SAFE" and review_at is None: review_at = s
            if no_face or r["status"] == "IDENTITY DRIFT":
                if drift_at is None: drift_at = s
            if wsc is not None and wsc < 80 and attr_at is None: attr_at = s
        rows[-1].update(review_at=review_at, drift_at=drift_at, attr_at=attr_at)
    return rows, images
