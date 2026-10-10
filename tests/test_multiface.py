"""Two faces in one frame: QA must compare the one that matches the reference, not the largest."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from faceprint import transforms as T
from faceprint.engine import FaceEngine
from faceprint.qa import Baseline

img = cv2.imread("samples/demo_face.jpg"); base = Baseline(FaceEngine(), img)
other = T.apply("Stylize", img, base.face.lm, 1.0)
# the stylised double is on the left AND larger (512px vs 400px) -> the old "largest face" rule would pick it
small = cv2.copyMakeBorder(cv2.resize(img, (400, 400)), 0, 112, 0, 0, cv2.BORDER_CONSTANT)
frame = np.hstack([other, small])
r = base.check(frame)
print("faces:", r["n_faces"], "cosine:", round(r["cosine"], 2), r["status"], "|", r["advice"][:1])
assert r["n_faces"] == 2 and r["cosine"] > 0.9 and r["status"] == "SAFE"
assert "faces in the frame" in r["advice"][0]
print("OK multiface")
