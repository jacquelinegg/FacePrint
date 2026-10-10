"""Builds demo_data/ (offline fallback): original, 6 SIMULATED campaign images (local OpenCV mutations),
policy, QA CSV, stress CSV, PDF report.  python scripts/make_demo_data.py [photo.jpg]
Use a photo of a CONSENTING person for the real demo; the bundled sample is only a placeholder.
In the pitch, label these images 'simulated'; use YouCam outputs (scripts/live_check.py) for the real ones."""
import os, sys, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, pandas as pd
from faceprint import transforms as T
from faceprint.engine import FaceEngine
from faceprint.qa import Baseline, stress_sweep
from faceprint.policy import Policy, evaluate
from faceprint.report import build

src = sys.argv[1] if len(sys.argv) > 1 else "samples/demo_face.jpg"
d = "demo_data"; os.makedirs(d, exist_ok=True)
img = cv2.imread(src); base = Baseline(FaceEngine(), img); lm = base.face.lm
cv2.imwrite(f"{d}/original.jpg", img)
def chain(*steps):
    im = img
    for n, s in steps: im = T.apply(n, im, lm, s)
    return im
camp = {"01_luxury_grade.jpg": chain(("Tone / lighting", 0.5)), "02_soft_glam.jpg": chain(("Skin smoothing", 0.5), ("Tone / lighting", 0.3)),
        "03_eyes_pop.jpg": chain(("Bigger eyes", 0.9), ("Fuller lips", 0.4)), "04_editorial_slim.jpg": chain(("Slimmer jaw", 1.0), ("Tone / lighting", 0.5)),
        "05_beauty_max.jpg": chain(("Combined beauty", 0.6)), "06_cyberpunk.jpg": chain(("Stylize", 1.0)),
        "07_smile_lipstick.jpg": chain(("Smile", 0.5), ("Lipstick", 0.8))}
os.makedirs(f"{d}/campaign", exist_ok=True); items, rows = [], []
for n, im in camp.items():
    cv2.imwrite(f"{d}/campaign/{n}", im); r = base.check(im); items.append({"name": n, "img": im, "result": r})
pol = Policy(talent="Demo model", campaign="Spring 2026 (simulated)", min_cosine=None,
             locked={"Face shape": 70, "Eyes": 70}); pol.save(f"{d}/policy.json")
for it in items:
    v, viol = evaluate(pol, it["result"]); rows.append({"image": it["name"], "similarity": round(it["result"]["cosine"], 3), "status": it["result"]["status"],
                                                        "policy": v, "issues": "; ".join(m for _, m in viol)})
pd.DataFrame(rows).to_csv(f"{d}/campaign_results.csv", index=False)
srows, _ = stress_sweep(base, img); pd.DataFrame(srows).to_csv(f"{d}/stress_test.csv", index=False)
print(build(f"{d}/faceprint_report.pdf", pol, img, items)["counts"]); print(pd.DataFrame(rows).to_string(index=False))
