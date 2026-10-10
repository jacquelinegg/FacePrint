"""Edge cases: no face, image without a face must not crash QA / policy / report."""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
from faceprint.engine import FaceEngine
from faceprint.qa import Baseline
from faceprint.policy import Policy, evaluate
from faceprint.report import build

img = cv2.imread("samples/demo_face.jpg"); base = Baseline(FaceEngine(), img)
blank = np.full((300, 300, 3), 127, np.uint8)
r = base.check(blank); assert r["status"] == "NO FACE" and r["n_faces"] == 0
pol = Policy.load("samples/policy.json"); v, viol = evaluate(pol, r); assert v == "NON-COMPLIANT"
build(os.path.join(tempfile.gettempdir(), "edge.pdf"), pol, img, [{"name": "blank.jpg", "img": blank, "result": r}, {"name": "same.jpg", "img": img, "result": base.check(img)}])
try: Baseline(FaceEngine(), blank); raise SystemExit("expected ValueError")
except ValueError: pass
print("OK edge cases")
