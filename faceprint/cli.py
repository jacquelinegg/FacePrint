import argparse, glob, json, os
import cv2, pandas as pd
from .engine import FaceEngine
from .qa import Baseline, stress_sweep

def main():
    p = argparse.ArgumentParser(prog="faceprint")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("qa", help="compare AI-generated images to the original")
    q.add_argument("original"); q.add_argument("variants", nargs="+"); q.add_argument("--out", default="out")
    s = sub.add_parser("stress", help="mutation stress test")
    s.add_argument("original"); s.add_argument("--out", default="out")
    c = sub.add_parser("calibrate", help="calibrate thresholds on <root>/<person>/*.jpg")
    c.add_argument("root"); c.add_argument("--max-people", type=int, default=200,
        help="identities to analyse (default 200; more = denser genuine histogram, slower)")
    y = sub.add_parser("youcam", help="YouCam attribute drift (needs YOUCAM_API_KEY; args = local paths or public URLs)")
    y.add_argument("original_url"); y.add_argument("variant_urls", nargs="+")
    l = sub.add_parser("lock", help="generate a YouCam effect and find its max identity-safe intensity")
    l.add_argument("original"); l.add_argument("--effect", choices=["aging", "hair", "makeup", "smile", "skin"], required=True)
    l.add_argument("--ref"); l.add_argument("--template", default="female_blunt_bob"); l.add_argument("--out", default="out")
    l.add_argument("--age", type=int, help="aging only: target res_age from the API series (below the person's age rejuvenates)")
    k = sub.add_parser("campaign", help="check a campaign against a talent policy and write a PDF report")
    k.add_argument("original"); k.add_argument("variants", nargs="+"); k.add_argument("--policy"); k.add_argument("--talent")
    k.add_argument("--campaign"); k.add_argument("--out", default="out")
    a = p.parse_args()
    if a.cmd == "campaign":
        from .policy import Policy
        from .report import build
        os.makedirs(a.out, exist_ok=True); orig = cv2.imread(a.original)
        pol = Policy.load(a.policy) if a.policy else Policy()
        pol.talent, pol.campaign = a.talent or pol.talent, a.campaign or pol.campaign
        base, items = Baseline(FaceEngine(), orig), []
        for path in sorted(f for pat in a.variants for f in glob.glob(pat)):
            im = cv2.imread(path); items.append({"name": os.path.basename(path), "img": im, "result": base.check(im)})
        info = build(f"{a.out}/faceprint_report.pdf", pol, orig, items)
        print(f"Report {info['report_id']}: {info['counts']}"); [print(f"  {n}: {v}") for n, v in info["rows"]]
        return
    if a.cmd == "lock":
        from .youcam import YouCamClient
        from .generators import generate
        from .lock import find_safe_intensity, find_safe_native
        from .generators import NATIVE
        os.makedirs(a.out, exist_ok=True); orig = cv2.imread(a.original)
        base, cl = Baseline(FaceEngine(), orig), YouCamClient()
        if a.effect in NATIVE:    # real API strength parameter, one call per level
            gen = generate(cl, a.effect, a.original, level=1.0)
            r = find_safe_native(base, orig, lambda lv: gen if lv == 1.0 else generate(cl, a.effect, a.original, level=lv))
        else:
            gen = generate(cl, a.effect, a.original, ref=a.ref, template_id=a.template, age_target=a.age)
            r = find_safe_intensity(base, orig, gen)
        cv2.imwrite(f"{a.out}/{a.effect}_full.jpg", gen); cv2.imwrite(f"{a.out}/{a.effect}_safe.jpg", r["safe_image"])
        if getattr(cl, "last_age", None) is not None: print(f"aging series -> res_age {cl.last_age:.0f}")
        pd.DataFrame(r["curve"]).to_csv(f"{a.out}/{a.effect}_curve.csv", index=False)
        print(pd.DataFrame(r["curve"]).to_string(index=False))
        print(f"\nFull effect: {r['full_effect']['status']} ({r['full_effect']['cosine']:.2f}); "
              f"max SAFE intensity = {r['safe_intensity']:.0%}, max REVIEW intensity = {r['review_intensity']:.0%}")
        return
    if a.cmd == "youcam":
        from .youcam import YouCamClient, compare_results
        cl = YouCamClient(); ref = cl.analyze_any(a.original_url)
        for u in a.variant_urls:
            r = compare_results(ref, cl.analyze_any(u)); print("\n" + u)
            print({k: round(v) for k, v in r["categories"].items()})
            for c in r["changes"]: print(f"  {c['category']}: {c['attribute']}: {c['before']} -> {c['after']}")
        return

    if a.cmd == "calibrate":
        from . import calibrate
        print(json.dumps(calibrate.run(a.root, max_people=a.max_people)["thresholds"], indent=2)); return
    os.makedirs(a.out, exist_ok=True)
    base = Baseline(FaceEngine(), cv2.imread(a.original))
    if a.cmd == "qa":
        rows = []
        for path in [f for pat in a.variants for f in glob.glob(pat)]:
            r = base.check(cv2.imread(path))
            rows.append({"image": os.path.basename(path), "status": r["status"], "cosine": round(r["cosine"], 3),
                         **{k: round(v) for k, v in r["categories"].items()}, "advice": "; ".join(r["advice"])})
        df = pd.DataFrame(rows); df.to_csv(f"{a.out}/qa_report.csv", index=False)
        print(df.drop(columns="advice").to_string(index=False))
    else:
        rows, imgs = stress_sweep(base, cv2.imread(a.original))
        for (n, st), im in imgs.items():
            cv2.imwrite(f"{a.out}/{n.replace(' ', '_').replace('/', '')}_{int(st*100):03d}.jpg", im)
        df = pd.DataFrame(rows); df.to_csv(f"{a.out}/stress.csv", index=False)
        print(df[["transform", "strength", "cosine", "status"]].to_string(index=False))

if __name__ == "__main__":
    main()
