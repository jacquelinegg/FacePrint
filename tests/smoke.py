"""Run: python tests/smoke.py   (needs internet on first run to fetch InsightFace models)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from faceprint.engine import FaceEngine
from faceprint.qa import Baseline
from faceprint import transforms as T

img = cv2.imread("samples/demo_face.jpg"); base = Baseline(FaceEngine(), img)
same = base.check(img); assert same["cosine"] > 0.99 and same["status"] == "SAFE"
mild = base.check(T.apply("Bigger eyes", img, base.face.lm, 0.5))
wild = base.check(T.apply("Stylize", img, base.face.lm, 1.0))
assert wild["cosine"] < mild["cosine"] < same["cosine"], (same["cosine"], mild["cosine"], wild["cosine"])
assert mild["categories"]["Eyes"] < same["categories"]["Eyes"]
print("OK", round(same["cosine"], 2), round(mild["cosine"], 2), round(wild["cosine"], 2), mild["categories"])
