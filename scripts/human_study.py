"""Mini human study: do people agree with the FACEPRINT score?
  python scripts/human_study.py make demo_data/original.jpg demo_data/campaign   -> human_study.html + human_study_key.json
  (send the HTML to 5-10 people; each downloads a CSV)
  python scripts/human_study.py analyze responses/*.csv                          -> agreement with the cosine score
Raters are blind (the key with scores stays with you)."""
import base64, glob, json, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, pandas as pd

def b64(img):
    return base64.b64encode(cv2.imencode(".jpg", cv2.resize(img, (360, int(360 * img.shape[0] / img.shape[1]))))[1]).decode()

def make(orig_path, folder):
    from faceprint.engine import FaceEngine
    from faceprint.qa import Baseline
    orig = cv2.imread(orig_path); base = Baseline(FaceEngine(), orig); key, cards = {}, []
    paths = sorted(glob.glob(os.path.join(folder, "*.*"))); random.Random(1).shuffle(paths)
    for i, p in enumerate(paths):
        r = base.check(cv2.imread(p)); key[f"P{i+1:02d}"] = {"file": os.path.basename(p), "cosine": r["cosine"]}
        cards.append(f'<div class=c><img src="data:image/jpeg;base64,{b64(orig)}"><img src="data:image/jpeg;base64,{b64(cv2.imread(p))}">'
                     f'<p><b>{i+1}.</b> Is the right image the same person as the left?</p>'
                     + "".join(f'<label><input type=radio name=P{i+1:02d} value={v}> {v}</label> ' for v in range(1, 6))
                     + "<small> (1 = definitely different, 5 = definitely the same person)</small></div>")
    html = ("<meta charset=utf-8><style>body{font-family:sans-serif;max-width:800px;margin:auto}.c{border-bottom:1px solid #ccc;padding:12px}"
            "img{width:46%;margin-right:2%}</style><h2>Same person? (blind rating)</h2>" + "".join(cards) +
            "<p><input id=n placeholder='your name/initials'> <button onclick=\"var r=['pair,rating'];document.querySelectorAll('input[type=radio]:checked')"
            ".forEach(x=>r.push(x.name+','+x.value));var a=document.createElement('a');a.href=URL.createObjectURL(new Blob([r.join('\\n')]));"
            "a.download='responses_'+document.getElementById('n').value+'.csv';a.click()\">Download my answers</button></p>")
    open("human_study.html", "w").write(html); json.dump(key, open("human_study_key.json", "w"), indent=1)
    print(f"{len(cards)} pairs -> human_study.html (key stays in human_study_key.json)")

def analyze(files):
    key = json.load(open("human_study_key.json"))
    df = pd.concat([pd.read_csv(f).assign(rater=f) for f in files]); m = df.groupby("pair")["rating"].agg(["mean", "std", "count"])
    m["cosine"] = [key[p]["cosine"] for p in m.index]; m["file"] = [key[p]["file"] for p in m.index]
    rho = m["mean"].rank().corr(m["cosine"].rank())
    print(m.sort_values("cosine", ascending=False).round(2).to_string())
    print(f"\n{df['rater'].nunique()} raters, {len(m)} pairs. Spearman correlation(human rating, FACEPRINT cosine) = {rho:.2f}")

if __name__ == "__main__":
    make(*sys.argv[2:4]) if sys.argv[1] == "make" else analyze(sys.argv[2:])
