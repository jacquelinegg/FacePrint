"""Attribute breakdown: robust to benign changes, sensitive to the edit that was actually made."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from faceprint import transforms as T
from faceprint.engine import FaceEngine
from faceprint.qa import Baseline

img = cv2.imread("samples/demo_face.jpg"); base = Baseline(FaceEngine(), img); lm = base.face.lm
cat = lambda im: base.check(im)["categories"]
jpeg = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 40])[1], 1)
small = cv2.resize(cv2.resize(img, None, fx=.6, fy=.6), (512, 512))
for name, im in [("jpeg40", jpeg), ("resize", small)]:
    c = cat(im); print(name, {k[:5]: round(v) for k, v in c.items()})
    assert all(v >= 85 for k, v in c.items() if k != "Skin tone"), (name, c)   # nothing should look "lost"

# (transform, strength, category that must drop below `limit`)
cases = [("Bigger eyes", 1.0, "Eyes", 60), ("Slimmer jaw", 0.6, "Face shape", 70), ("Fuller lips", 0.6, "Lips", 75),
         ("Smile", 0.6, "Expression", 75), ("Head turn", 0.6, "Head pose", 80), ("Lipstick", 1.0, "Lips", 92),
         ("Tone / lighting", 0.6, "Skin tone", 70)]
for name, s, target, limit in cases:
    c = cat(T.apply(name, img, lm, s)); print(f"{name:16s}{s}: {target} = {c[target]:.0f} (< {limit})")
    assert c[target] < limit, (name, c)
print("OK attributes")
