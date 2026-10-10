"""Identity Lock with real ArcFace; the YouCam call is mocked (generator returns a stylized face)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from faceprint import generators, transforms as T
from faceprint.engine import FaceEngine
from faceprint.qa import Baseline
from faceprint.lock import find_safe_intensity, find_safe_native

img = cv2.imread("samples/demo_face.jpg"); base = Baseline(FaceEngine(), img)
calls = []

class FakeClient:
    def run_generative(self, base_url, body, age=None):
        calls.append((base_url.rsplit("/", 1)[-1], body, age))
        return T.apply("Stylize", img, base.face.lm, 1.0)

gen = generators.generate(FakeClient(), "aging", "http://x/o.jpg", age_target=25)
eff, body, age = calls[-1]
assert eff == "aging" and body == {"src_file_url": "http://x/o.jpg"}, body   # v2 flat shape, no request_id
r = find_safe_intensity(base, img, gen)
for row in r["curve"]: print(row["intensity"], round(row["cosine"], 2), row["status"])
assert r["curve"][0]["status"] == "SAFE" and r["curve"][-1]["status"] != "SAFE"
assert 0 < r["safe_intensity"] < 1 and r["review_intensity"] >= r["safe_intensity"]
print("OK safe<=", r["safe_intensity"], "review<=", r["review_intensity"])

# --- native-strength effects (skin) and the verified request shapes of smile / skin / makeup / hair ---
generators.generate(FakeClient(), "smile", "http://x/o.jpg")
assert calls[-1][:2] == ("ai-smile", {"src_file_url": "http://x/o.jpg", "expression_type": "closed_mouth_smile"})
generators.generate(FakeClient(), "skin", "http://x/o.jpg", level=0.5)
eff, sk, _ = calls[-1]
assert eff == "skin-simulation" and sk["src_file_url"] == "http://x/o.jpg" and sk["acne"] == 0.5 and len(sk) == 11, sk
generators.generate(FakeClient(), "makeup", "http://x/o.jpg", ref="http://x/ref.png")
assert calls[-1][:2] == ("mu-transfer", {"src_file_url": "http://x/o.jpg", "ref_file_url": "http://x/ref.png"})
generators.generate(FakeClient(), "hair", "http://x/o.jpg", template_id="female_blunt_bob")
assert calls[-1][:2] == ("hair-transfer", {"src_file_url": "http://x/o.jpg", "template_id": "female_blunt_bob",
                                       "hair_color": "src"})
print("OK request shapes verified:", [c[0] for c in calls])

def make(lv):
    calls.append(("skin", {"level": lv})); return T.apply("Stylize", img, base.face.lm, lv)
n = find_safe_native(base, img, make)
assert [c[1]["level"] for c in calls if c[0] == "skin"][:1] == [0.1] and n["native"]      # finer low end
assert n["curve"][0]["cosine"] > 0.99 and n["safe_intensity"] <= n["review_intensity"] <= 1
print("OK native + request shapes: safe<=", n["safe_intensity"], "review<=", n["review_intensity"])

