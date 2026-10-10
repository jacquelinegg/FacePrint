"""Download LFW (~173 MB) and arrange <out>/<person>/*.jpg for `python -m faceprint calibrate`.
    python scripts/get_lfw.py --people 300 --out data/lfw     (needs internet; UNTESTED here)"""
import argparse, os, shutil, tarfile, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("--people", type=int, default=300); ap.add_argument("--out", default="data/lfw"); a = ap.parse_args()
os.makedirs("data", exist_ok=True); tgz = "data/lfw.tgz"
if not os.path.exists(tgz):
    print("downloading LFW..."); urllib.request.urlretrieve("http://vis-www.cs.umass.edu/lfw/lfw.tgz", tgz)
tarfile.open(tgz).extractall("data/lfw_raw")
root = next(os.path.join("data/lfw_raw", p) for p in os.listdir("data/lfw_raw"))
people = sorted(p for p in os.listdir(root) if len(os.listdir(os.path.join(root, p))) >= 2)[: a.people]
for p in people: shutil.copytree(os.path.join(root, p), os.path.join(a.out, p), dirs_exist_ok=True)
print(f"{len(people)} people -> {a.out}\nnext: python -m faceprint calibrate {a.out}")
