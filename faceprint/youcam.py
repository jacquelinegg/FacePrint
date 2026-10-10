"""YouCam (Perfect Corp) Face Attribute Analysis: async client + attribute-drift comparison.
The API key is read from the YOUCAM_API_KEY environment variable (never hard-code it).
Inputs: a public image URL (verified) or local bytes via the File API (/s2s/v2.0/file -> presigned PUT -> file_id).
Photo requirements (docs): single frontal face, face width > 60% of the image, 320..4096 px, < 10 MB, jpg/png."""
import json, math, os, time, urllib.error, urllib.request
import cv2, numpy as np
from .engine import FaceEngine

BASE = "https://yce-api-01.makeupar.com/s2s/v2.0/task/face-attr-analysis"
FEATURES = ["gender", "age", "faceShape", "cheekbones", "horizontalThird", "verticalFifth", "faceAspectRatio",
            "eyeShape", "eyeSize", "eyeAngle", "eyeDistance", "eyelid", "eyeAspectRatio",
            "eyeHeightToEyebrowDistance", "eyebrowShape", "eyebrowThickness", "eyebrowDistance",
            "eyebrowShortness", "eyebrowArch", "lipShape", "upperLipToLowerLip", "noseWidth", "noseLength",
            "noseAspectRatio", "noseWidthToMouthWidth", "noseToLipToChin"]

FILE_API = "https://yce-api-01.makeupar.com/s2s/v2.0/file"
S2S = "https://yce-api-01.makeupar.com/s2s"
TEMPLATE_TPL = S2S + "/v2.1/task/template/{feature}"

class YouCamError(RuntimeError): pass

def prepare_image(data, max_side=4096, max_bytes=10_000_000):
    """Validate/resize to API limits. Returns (bytes, file_name, content_type)."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise YouCamError("Not a readable image")
    h, w = img.shape[:2]
    if min(h, w) < 320: raise YouCamError("Image must be at least 320 px (error_below_min_image_size)")
    is_png = data[:4] == b"\x89PNG"
    if max(h, w) > max_side or len(data) > max_bytes:
        k = min(1.0, max_side / max(h, w))
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
        data, is_png = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(), False
    return data, ("image.png" if is_png else "image.jpg"), ("image/png" if is_png else "image/jpg")

def crop_to_face(image_bytes, pad_x=0.30, pad_y=0.50, min_side=360, straighten=True):
    """Crop around the detected face before upload so YouCam's face-size check passes
    (`error_face_position_too_small` is the most common task failure otherwise).
    Roll is straightened with the 5 keypoints when possible - `error_face_not_forward_facing`
    / `error_large_face_angle` on a headshot is usually a tilted camera.
    Returns jpeg bytes; falls back to the original bytes when no face is detected."""
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None: return image_bytes
    face, _ = FaceEngine().analyze(img)
    if face is None: return image_bytes
    x1, y1, x2, y2 = face.bbox
    h, w = img.shape[:2]
    px, py = pad_x * (x2 - x1), pad_y * (y2 - y1)
    crop = img[max(0, int(y1 - py)):min(h, int(y2 + py)), max(0, int(x1 - px)):min(w, int(x2 + px))]
    if crop.size == 0: return image_bytes
    if straighten and getattr(face, "kps", None) is not None:
        le, re = face.kps[0], face.kps[1]
        roll = float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0])))
        if abs(roll) > 0.5:
            m = cv2.getRotationMatrix2D((crop.shape[1] / 2, crop.shape[0] / 2), roll, 1.0)
            crop = cv2.warpAffine(crop, m, (crop.shape[1], crop.shape[0]), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
    if min(crop.shape[:2]) < min_side:
        k = min_side / min(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes() if ok else image_bytes

def _age_series(results):
    """Aging returns `results.output[]` = [{res_age, url}, ...]: a whole age range in ONE call.
    Other effects return a single `results.url`. -> [(res_age, url)] or []."""
    if not isinstance(results, dict): return []
    out = results.get("output")
    if not isinstance(out, list): return []
    return [(float(i["res_age"]), i["url"]) for i in out
            if isinstance(i, dict) and str(i.get("url", "")).startswith("http") and i.get("res_age") is not None]

def _pick_age(series, age):
    """The series entry whose res_age is closest to the requested age."""
    return min(series, key=lambda t: abs(t[0] - age))[1] if series else None

def _find_url(o):
    """First download URL inside a task result payload. Handles the single-URL shape
    ({results:{url}}) and aging's series ({results:{output:[{url,res_age}, ...]}} -> most extreme age)."""
    if isinstance(o, dict):
        out = o.get("output")
        if isinstance(out, list):
            items = [i for i in out if isinstance(i, dict) and str(i.get("url", "")).startswith("http")]
            if items: return max(items, key=lambda i: i.get("res_age") or 0)["url"]
        if str(o.get("url", "")).startswith("http"): return o["url"]
        for v in o.values():
            if u := _find_url(v): return u
    if isinstance(o, list):
        for v in o:
            if u := _find_url(v): return u
    return None

class YouCamClient:
    def __init__(self, api_key=None, interval=2.0, max_attempts=150, strictness="high"):
        self.key = api_key or os.environ.get("YOUCAM_API_KEY")
        if not self.key:
            raise YouCamError("Set the YOUCAM_API_KEY environment variable")
        self.interval, self.max_attempts, self.strictness = interval, max_attempts, strictness
        self.last_results = None

    def _req(self, method, url, body=None, headers=None, raw=None):
        h = {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"} if headers is None else headers
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method=method, data=data, headers=h), timeout=60) as r:
                txt = r.read()
                return json.loads(txt) if txt and raw is None else {}
        except urllib.error.HTTPError as e:
            raise YouCamError(f"HTTP {e.code}: {e.read()[:300]!r}") from None

    def upload(self, image_bytes):
        """File API: register metadata, PUT bytes to the presigned URL, return file_id."""
        data, name, ctype = prepare_image(image_bytes)
        meta = self._req("POST", FILE_API, {"files": [{"content_type": ctype, "file_name": name, "file_size": len(data)}]})
        try:
            f = meta["data"]["files"][0]; fid, put = f["file_id"], f["requests"][0]
        except (KeyError, IndexError, TypeError):
            raise YouCamError(f"Unexpected File API response: {str(meta)[:300]}") from None
        self._req(put.get("method", "PUT"), put["url"], headers=put.get("headers", {}), raw=data)  # presigned: no Bearer
        return fid

    def analyze(self, image_url=None, file_id=None):
        """Start the task (public URL or file_id), poll until done, return the `results` dict.
        Verified body shape (docs.perfectcorp.com face-attr-analysis): flat top-level
        `features` + `face_angle_strictness_level` + one of `src_file_url` / `src_file_id`."""
        body = {"features": FEATURES, "face_angle_strictness_level": self.strictness}
        if image_url:     body["src_file_url"] = image_url
        else:             body["src_file_id"] = file_id
        return self.run_task(BASE, body)

    def run_task(self, base, body):
        """POST a task to `base`, poll `base/{task_id}` until success; return `results`."""
        task = self._req("POST", base, body)
        tid = (task.get("data") or {}).get("task_id")
        if not tid: raise YouCamError(f"No task_id in response: {str(task)[:300]}")
        for _ in range(self.max_attempts):
            d = self._req("GET", f"{base}/{tid}").get("data", {})
            if d.get("task_status") == "success":
                self.last_results = d["results"]; return d["results"]
            if d.get("task_status") == "error": raise YouCamError(f"Task failed: {d.get('error')}")
            time.sleep(self.interval)
        raise YouCamError("Timed out while polling")

    def analyze_bytes(self, image_bytes):
        return self.analyze(file_id=self.upload(crop_to_face(image_bytes)))

    def image(self, url, timeout=120):
        """Download a task result image (presigned URL, no Authorization) and decode it to BGR."""
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = r.read()
        except Exception as e:
            raise YouCamError(f"Cannot download result image: {e}") from None
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None: raise YouCamError(f"Result URL did not return a readable image: {url[:120]}")
        return img

    def analyze_any(self, src):
        """src = http(s) URL or local path."""
        return self.analyze(image_url=src) if src.startswith("http") else self.analyze_bytes(open(src, "rb").read())

    def run_generative(self, base, body, age=None):
        """POST a generative task (aging / hair / makeup / smile / skin), poll until success,
        download and decode the result image (BGR). Used by faceprint.generators.
        `age` selects an entry of aging's `output[]` series (res_age); None keeps the strongest."""
        task = self._req("POST", base, body)
        tid = (task.get("data") or {}).get("task_id")
        if not tid: raise YouCamError(f"No task_id in response: {str(task)[:300]}")
        for _ in range(self.max_attempts):
            d = self._req("GET", f"{base}/{tid}").get("data", {})
            if d.get("task_status") == "success":
                self.last_results = d.get("results")
                series = _age_series(d.get("results"))
                self.last_age = None
                if series:
                    self.last_age = min(series, key=lambda t: abs(t[0] - age))[0] if age is not None \
                                    else max(series)[0]
                url = (_pick_age(series, age) if age is not None else _find_url(d.get("results")))
                if not url: raise YouCamError(f"No result url in task data: {str(d)[:300]}")
                return self.image(url)
            if d.get("task_status") == "error":
                raise YouCamError(f"Task failed: {d.get('error')} - {d.get('error_message') or ''}".strip(" -"))
            time.sleep(self.interval)
        raise YouCamError("Timed out while polling")

    def templates(self, feature="hair-transfer", page_size=20, starting_token=None):
        """Predefined YouCam style templates, e.g. GET /s2s/v2.1/task/template/hair-transfer (not billable)."""
        q = f"?page_size={page_size}" + (f"&starting_token={starting_token}" if starting_token else "")
        return ((self._req("GET", TEMPLATE_TPL.format(feature=feature) + q)).get("data") or {}).get("templates", [])

# ---- comparison -------------------------------------------------------------------------------
# kind: cat = categorical label, ratio = [1, x] ratio (relative tol), pts = list of percentages (abs tol), num = number
SPEC = {
 "Face shape": [("faceshape", "cat", 0), ("cheekbone.overrall", "cat", 0), ("cheekbone.overall", "cat", 0),
                ("facialratio.face_aspect_ratio", "ratio", .08),
                ("facialratio.horizontal_third", "pts", 4), ("facialratio.vertical_fifth", "pts", 4)],
 "Eyes": [(f"eyelid.{k}", "cat", 0) for k in ("left_eyelid", "right_eyelid", "left_angle", "right_angle",
                                              "left_shape", "right_shape", "setting", "size")] +
         [("facialratio.left_eye_aspect_ratio", "ratio", .10), ("facialratio.right_eye_aspect_ratio", "ratio", .10),
          ("facialratio.overall_eye_height_to_eyebrow_distance", "ratio", .10)],
 "Brows": [(f"eyebrow.{k}", "cat", 0) for k in ("gap", "left_shape", "right_shape", "left_body_thickness",
           "left_tail_thickness", "left_thinness", "right_body_thickness", "right_tail_thickness",
           "right_thinness", "left_shortness", "right_shortness")] +
          [("facialratio.left_eyebrow_arch_to_eyebrow_width", "ratio", .10),
           ("facialratio.right_eyebrow_arch_to_eyebrow_width", "ratio", .10)],
 "Lips": [("lipshape", "cat", 0), ("facialratio.upper_lip_to_lower_lip", "ratio", .10)],
 "Nose": [("nose.width", "cat", 0), ("nose.length", "cat", 0), ("facialratio.nose_aspect_ratio", "ratio", .08),
          ("facialratio.nose_width_to_mouth_width", "ratio", .08), ("facialratio.nose_to_lip_to_chin", "ratio", .08)],
 "Age / gender": [("agegender.gender", "cat", 0), ("agegender.age", "num", 5)],
}

def _get(res, path):
    for p in path.split("."):
        res = res.get(p) if isinstance(res, dict) else None
        if res is None: return None
    return res

def _score(kind, tol, a, b):
    if kind == "cat":
        a, b = (", ".join(a), ", ".join(b)) if isinstance(a, list) else (a, b)
        return (100.0 if a == b else 0.0), a, b
    if kind == "ratio": d, va, vb = abs(b[1] / b[0] - a[1] / a[0]) / (a[1] / a[0]), round(a[1] / a[0], 2), round(b[1] / b[0], 2)
    elif kind == "pts": d, va, vb = sum(abs(x - y) for x, y in zip(a, b)) / len(a), a, b
    else: d, va, vb = abs(b - a), a, b
    return 100 * math.exp(-0.5 * d / tol), va, vb

def compare_results(ref, new):
    """Compare two `results` dicts -> per-category score + list of concrete before/after changes."""
    cats, changes = {}, []
    for cat, items in SPEC.items():
        scores = []
        for path, kind, tol in items:
            a, b = _get(ref, path), _get(new, path)
            if a is None or b is None: continue
            s, va, vb = _score(kind, tol, a, b); scores.append(s)
            if s < 60: changes.append({"category": cat, "attribute": path.split(".", 1)[-1].replace("_", " "),
                                       "before": va, "after": vb})
        if scores: cats[cat] = sum(scores) / len(scores)
    return {"categories": cats, "changes": changes}
