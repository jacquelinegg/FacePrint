"""Empirical calibration on a labelled folder: <root>/<person>/*.jpg (e.g. LFW).
Computes genuine (same person) vs impostor (different people) cosine distributions and derives:
  review = impostor 99th percentile (~1% false-accept rate), safe = median of genuine pairs."""
import glob, itertools, json, os, random
import cv2, numpy as np
from .engine import FaceEngine
from .qa import CAL_PATH

def run(root, max_people=200, seed=0):
    eng, rnd, people = FaceEngine(), random.Random(seed), {}
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        embs = []
        for p in glob.glob(os.path.join(d, "*.*")):
            img = cv2.imread(p)
            f = eng.analyze(img)[0] if img is not None else None
            if f: embs.append(f.embedding)
        if embs: people[os.path.basename(d)] = embs
        if len(people) >= max_people: break
    gen = [float(a @ b) for e in people.values() if len(e) > 1 for a, b in itertools.combinations(e, 2)]
    names, imp = list(people), []
    for _ in range(min(5000, len(names) * 25)):
        a, b = rnd.sample(names, 2)
        imp.append(float(rnd.choice(people[a]) @ rnd.choice(people[b])))
    if len(gen) < 10 or len(imp) < 10:
        raise SystemExit("Need >=10 genuine and impostor pairs (people with 2+ photos each)")
    th = {"safe": float(np.median(gen)), "review": float(np.percentile(imp, 99)),
          "source": f"calibrated on {len(people)} identities ({len(gen)} genuine / {len(imp)} impostor pairs)"}
    bins = np.linspace(-0.2, 1.0, 25)
    out = {"thresholds": th, "bins": bins.tolist(),
           "genuine_hist": np.histogram(gen, bins)[0].tolist(),
           "impostor_hist": np.histogram(imp, bins)[0].tolist()}
    json.dump(out, open(CAL_PATH, "w"), indent=2)
    return out
