"""Offline test of the YouCam comparison using the sample response (no network, no key)."""
import copy, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from faceprint.youcam import compare_results

ref = json.load(open("samples/youcam_sample.json"))
assert all(v == 100 for v in compare_results(ref, ref)["categories"].values())
new = copy.deepcopy(ref)
new["faceshape"] = "Oval"; new["nose"]["width"] = "Average"; new["eyelid"]["size"] = "Large"
new["facialratio"]["upper_lip_to_lower_lip"] = [1, 1.6]
r = compare_results(ref, new)
for c in r["changes"]: print(c)
assert {c["attribute"] for c in r["changes"]} >= {"faceshape", "width", "size", "upper lip to lower lip"}
print("OK", {k: round(v) for k, v in r["categories"].items()})

# --- mocked end-to-end flow: File API -> presigned PUT -> task -> poll (no network) ---
import io, json as _j, urllib.request as ur
import cv2, numpy as np
from faceprint.youcam import YouCamClient
calls = []
class R(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): pass
def fake(req, timeout=0):
    calls.append((req.get_method(), req.full_url, dict(req.header_items())))
    u = req.full_url
    if u.endswith("/s2s/v2.0/file"):
        body = _j.loads(req.data); assert body["files"][0]["file_size"] > 0
        return R(_j.dumps({"status": 200, "data": {"files": [{"file_id": "FID", "requests": [
            {"method": "PUT", "url": "https://s3.example/put", "headers": {"Content-Type": "image/jpg"}}]}]}}).encode())
    if u == "https://s3.example/put": assert req.get_method() == "PUT"; return R(b"")
    if req.get_method() == "POST":
        b = _j.loads(req.data); assert b["src_file_id"] == "FID" and b["features"]
        return R(_j.dumps({"status": 200, "data": {"task_id": "T1"}}).encode())
    n = sum(1 for c in calls if c[0] == "GET")
    st = {"task_status": "running"} if n < 2 else {"task_status": "success", "results": ref}
    return R(_j.dumps({"status": 200, "data": st}).encode())
ur.urlopen = fake
img = cv2.imencode(".jpg", np.full((400, 400, 3), 128, np.uint8))[1].tobytes()
res = YouCamClient("test-key", interval=0).analyze_bytes(img)
assert res == ref and "Authorization" not in calls[1][2] and "Authorization" in calls[0][2]
print("mock flow OK:", [c[0] for c in calls])
