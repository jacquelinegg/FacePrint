"""Identity Lock: find the strongest version of an AI effect that keeps the person recognisable.
Effects without a strength parameter are dialled by blending generated image with the original
(alpha 0 = original, 1 = full effect) and measuring identity at each step."""
import cv2, numpy as np

def blend(orig, gen, a):
    gen = cv2.resize(gen, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_AREA)
    return cv2.addWeighted(orig, 1 - a, gen, a, 0)

def find_safe_intensity(baseline, orig, gen, steps=11):
    """Return curve + largest contiguous alpha that stays SAFE / not DRIFT, plus the images."""
    curve, imgs = [], {}
    for a in np.linspace(0, 1, steps):
        im = blend(orig, gen, float(a)); r = baseline.check(im)
        curve.append({"intensity": round(float(a), 2), "cosine": r["cosine"], "status": r["status"]})
        imgs[round(float(a), 2)] = im
    def limit(ok):
        best = 0.0
        for row in curve:
            if not ok(row["status"]): break
            best = row["intensity"]
        return best
    safe, review = limit(lambda s: s == "SAFE"), limit(lambda s: s in ("SAFE", "REVIEW"))
    return {"curve": curve, "safe_intensity": safe, "review_intensity": review,
            "safe_image": imgs[safe], "full_effect": baseline.check(gen), "images": imgs}

def find_safe_native(baseline, orig, make, levels=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0)):
    """Same result shape as find_safe_intensity, but the strength is a NATIVE API parameter:
    make(level) -> image generated at that level (level 0 is the original, no API call is made for it).
    Costs one API call per non-zero level; finer near the low end where the SAFE boundary typically sits."""
    curve, imgs = [], {}
    for a in levels:
        im = orig if a == 0 else make(float(a)); r = baseline.check(im)
        curve.append({"intensity": round(float(a), 2), "cosine": r["cosine"], "status": r["status"]})
        imgs[round(float(a), 2)] = im
    def limit(ok):
        best = 0.0
        for row in curve:
            if not ok(row["status"]): break
            best = row["intensity"]
        return best
    safe, review = limit(lambda s: s == "SAFE"), limit(lambda s: s in ("SAFE", "REVIEW"))
    top = round(float(levels[-1]), 2)
    return {"curve": curve, "safe_intensity": safe, "review_intensity": review, "safe_image": imgs[safe],
            "full_effect": baseline.check(imgs[top]), "images": imgs, "native": True}
