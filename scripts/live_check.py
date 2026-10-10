"""PROOF of real YouCam integration. Runs the full flow against the live API and writes a reproducible
evidence log (no API key stored).   export YOUCAM_API_KEY=...
    python scripts/live_check.py photo.jpg [--effects aging hair] [--public-url https://...] [--ref-look https://...]
Output: evidence/<timestamp>/ evidence.json, raw_*.json, generated images, curve CSV, report PDF.
The raw_*.json files show the REAL response shapes - if a step fails, send me that file and the error."""
import argparse, datetime, hashlib, json, os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, pandas as pd
from faceprint.engine import FaceEngine
from faceprint.qa import Baseline, thresholds
from faceprint.youcam import YouCamClient, compare_results
from faceprint.generators import generate
from faceprint.lock import find_safe_intensity
from faceprint.policy import Policy
from faceprint.report import build

ap = argparse.ArgumentParser(); ap.add_argument("photo"); ap.add_argument("--effects", nargs="*", default=["aging", "hair", "smile"])
ap.add_argument("--public-url"); ap.add_argument("--ref-look"); ap.add_argument("--template", default="female_blunt_bob")
a = ap.parse_args()
out = f"evidence/{datetime.datetime.now():%Y%m%d_%H%M%S}"; os.makedirs(out, exist_ok=True)
log = {"started": datetime.datetime.now().isoformat(), "photo_sha256": hashlib.sha256(open(a.photo, "rb").read()).hexdigest(),
       "thresholds": thresholds(), "steps": []}
cl, orig = YouCamClient(strictness="low"), cv2.imread(a.photo)   # "low": real-world and AI shots are rarely perfectly level
photo_bytes = open(a.photo, "rb").read()

def step(name, fn):
    t = time.time()
    try:
        r = fn(); log["steps"].append({"step": name, "ok": True, "seconds": round(time.time() - t, 1)}); print("OK  ", name); return r
    except Exception as e:
        log["steps"].append({"step": name, "ok": False, "error": f"{type(e).__name__}: {e}"[:600]})
        print("FAIL", name, "-", str(e)[:300]); open(f"{out}/error_{name}.txt", "w").write(traceback.format_exc()); return None

def dump(name):
    if cl.last_results is not None: json.dump(cl.last_results, open(f"{out}/raw_{name}.json", "w"), indent=2)

base = step("local_arcface_baseline", lambda: Baseline(FaceEngine(), orig))
ref_attr = step("youcam_upload_and_attributes", lambda: cl.analyze_bytes(photo_bytes)); dump("attributes_original")
if a.public_url:
    step("youcam_attributes_via_public_url", lambda: cl.analyze(image_url=a.public_url)); dump("attributes_url")
items = []
for eff in a.effects if base else []:
    gen = step(f"youcam_generate_{eff}", lambda e=eff: generate(cl, e, photo_bytes, ref=a.ref_look, template_id=a.template)); dump(f"generate_{eff}")
    if gen is None: continue
    cv2.imwrite(f"{out}/{eff}_full.jpg", gen)
    gen_attr = step(f"youcam_attributes_{eff}", lambda g=gen: cl.analyze_bytes(cv2.imencode(".jpg", g)[1].tobytes()))
    changes = compare_results(ref_attr, gen_attr)["changes"] if ref_attr and gen_attr else []
    lock = step(f"identity_lock_{eff}", lambda g=gen: find_safe_intensity(base, orig, g))
    if lock:
        pd.DataFrame(lock["curve"]).to_csv(f"{out}/{eff}_curve.csv", index=False); cv2.imwrite(f"{out}/{eff}_safe.jpg", lock["safe_image"])
        log[f"{eff}_summary"] = {"full_effect_cosine": round(lock["full_effect"]["cosine"], 3), "full_effect_status": lock["full_effect"]["status"],
                                 "max_safe_intensity": lock["safe_intensity"], "max_review_intensity": lock["review_intensity"],
                                 "yc_attribute_changes": changes}
        items.append({"name": f"{eff}_full.jpg", "img": gen, "result": lock["full_effect"], "yc_changes": changes})
        items.append({"name": f"{eff}_identity_locked_{lock['safe_intensity']:.0%}.jpg", "img": lock["safe_image"], "result": base.check(lock["safe_image"])})
if items:
    info = step("pdf_report", lambda: build(f"{out}/report.pdf", Policy(talent="Live check", campaign="YouCam live test"), orig, items))
    log["report_id"] = info and info["report_id"]
log["finished"] = datetime.datetime.now().isoformat(); log["all_ok"] = all(s["ok"] for s in log["steps"])
json.dump(log, open(f"{out}/evidence.json", "w"), indent=2)
print(f"\n{'ALL STEPS OK' if log['all_ok'] else 'SOME STEPS FAILED'} -> {out}/evidence.json")
