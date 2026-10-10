import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from faceprint import transforms as T
from faceprint.engine import FaceEngine
from faceprint.qa import Baseline
from faceprint.policy import Policy, evaluate
from faceprint.report import build

img = cv2.imread("samples/demo_face.jpg"); base = Baseline(FaceEngine(), img); pol = Policy.load("samples/policy.json")
items = [{"name": n, "img": im, "result": base.check(im)} for n, im in [
    ("luxury_campaign.jpg", T.apply("Tone / lighting", img, base.face.lm, 0.6)),
    ("big_eyes.jpg", T.apply("Bigger eyes", img, base.face.lm, 1.0)),
    ("cyberpunk.jpg", T.apply("Stylize", img, base.face.lm, 1.0))]]
verdicts = [evaluate(pol, i["result"])[0] for i in items]; print(verdicts)
assert verdicts[0] == "COMPLIANT" and verdicts[2] == "NON-COMPLIANT"
info = build(os.path.join(tempfile.gettempdir(), "report.pdf"), pol, img, items); print(info["report_id"], os.path.getsize(os.path.join(tempfile.gettempdir(), "report.pdf")), "bytes")
