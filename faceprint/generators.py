"""YouCam generative effects + an identity-locking intensity sweep on top of them.

Endpoints and request bodies are verified against docs.perfectcorp.com (v2 run-task APIs,
BasicRunTaskV2 flat shape: `src_file_id`/`src_file_url` + effect params) and against the live API:
  aging     POST /s2s/v2.0/task/aging            src                       -> results.output[{url,res_age}]
  hair      POST /s2s/v2.1/task/hair-transfer    src + ONE of template_id / ref_file_id / ref_file_url [+ hair_color]
  makeup    POST /s2s/v2.0/task/mu-transfer      src + ONE of ref_file_id / ref_file_url
  smile     POST /s2s/v2.0/task/ai-smile         src + expression_type (required)
  skin      POST /s2s/v2.0/task/skin-simulation  src + per-concern 0..1 (at least one > 0)
Each image is auto-framed to the endpoint's file specs (crop to face + pixel limits) before upload,
because out-of-spec framing fails the task server-side and wastes units."""
import os
import cv2, numpy as np
from .youcam import YouCamClient, YouCamError
from .engine import FaceEngine

BASE = "https://yce-api-01.makeupar.com"
SKIN_CONCERNS = {
    "wrinkle": "Wrinkles", "pores": "Pores", "redness": "Redness", "acne": "Acne",
    "oiliness": "Oiliness", "eye_bags": "Eye bags", "dark_circle": "Dark circles",
    "spots": "Spots", "texture": "Texture", "radiance": "Radiance",
}
SMILE_TYPE = "closed_mouth_smile"        # other enum value: smile_with_teeth_visible

# effect -> (endpoint, max long side px, min short side px) - straight from each endpoint's File Specs
EFFECTS = {
    "aging":  ("/s2s/v2.0/task/aging",            4096, 320),
    "hair":   ("/s2s/v2.1/task/hair-transfer",    1024, 480),
    "makeup": ("/s2s/v2.0/task/mu-transfer",      1024, 480),
    "smile":  ("/s2s/v2.0/task/ai-smile",         4096, 480),
    "skin":   ("/s2s/v2.0/task/skin-simulation",  2560, 480),
}
NATIVE = {"skin"}      # only skin-simulation exposes a native strength axis; the other effects
                       # are generated once and identity-locked by blending (see faceprint.lock)
DEFAULT_TEMPLATE = "female_blunt_bob"   # real template_id from GET /s2s/v2.1/task/template/hair-transfer
TEMPLATE_API = "/s2s/v2.1/task/template/hair-transfer"


def _src(client, x, prefix):
    """File API for local bytes (src_file/ref_file), or pass a public URL through as *_file_url."""
    if x is None: return {}
    if isinstance(x, (bytes, bytearray)): return {f"{prefix}_id": client.upload(bytes(x))}
    return {f"{prefix}_url": x}

def _fit(img_bgr, max_long, min_short, pad_x=0.30, pad_y=0.50):
    """Crop to the detected face and rescale to the endpoint's file specs (face > 60% of the frame,
    pixel limits). Returns (fitted BGR image, crop rect [x, y, w, h] in the original frame) so the
    generated face can be composited back into the original photo afterwards."""
    face, _ = FaceEngine().analyze(img_bgr)
    if face is None: return img_bgr, (0, 0, img_bgr.shape[1], img_bgr.shape[0])
    x1, y1, x2, y2 = [float(v) for v in face.bbox]
    h, w = img_bgr.shape[:2]
    px, py = pad_x * (x2 - x1), pad_y * (y2 - y1)
    xa, ya = max(0, int(x1 - px)), max(0, int(y1 - py))
    xb, yb = min(w, int(x2 + px + 0.5)), min(h, int(y2 + py + 0.5))
    crop = img_bgr[ya:yb, xa:xb]
    if crop.size == 0: return img_bgr, (0, 0, w, h)
    if min(crop.shape[:2]) < min_short:
        k = min_short / min(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    if max(crop.shape[:2]) > max_long:
        k = max_long / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return crop, (xa, ya, xb - xa, yb - ya)

def _composite(orig, gen, rect):
    """Paste the generated face-crop back into the original framing so blending/lock work in the
    original pixels (and the UI shows the original photo with the AI-generated face)."""
    if gen is None: return orig
    rx, ry, rw, rh = rect
    if gen.shape[:2] == orig.shape[:2] or rw <= 0 or rh <= 0: return gen
    resized = cv2.resize(gen, (rw, rh), interpolation=cv2.INTER_AREA if gen.shape[1] > rw else cv2.INTER_CUBIC)
    out = orig.copy()
    out[ry:ry + rh, rx:rx + rw] = resized
    return out

def generate(client, effect, src, ref=None, template_id=None, expression_type=None, hair_color="src",
             level=1.0, age_target=None):
    """Run one generative YouCam effect and return the result composited back into the original
    photo (same frame -> the identity-lock blend works in original pixels).
    `src`/`ref` are local bytes (uploaded through the File API) or a public image URL.
    `age_target` (aging only): pick the series entry with that res_age - below the person's age
    rejuvenates, above ages them. `client.last_age` reports what the API actually produced."""
    path, max_long, min_short = EFFECTS[effect]
    if effect == "makeup" and ref is None:
        raise YouCamError("makeup needs a reference look (local bytes or a public URL)")
    src = _read(src)
    if isinstance(src, np.ndarray):                 # accept a decoded BGR image as well as bytes/URL
        ok, buf = cv2.imencode(".jpg", src, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok: raise YouCamError("Cannot encode the source image")
        src = buf.tobytes()
    orig, rect = None, None
    if isinstance(src, (bytes, bytearray)):
        orig = cv2.imdecode(np.frombuffer(src, np.uint8), cv2.IMREAD_COLOR)
        src, rect = _fit(orig, max_long, min_short)
        ok, buf = cv2.imencode(".jpg", src, [cv2.IMWRITE_JPEG_QUALITY, 92])
        src = buf.tobytes()
    if isinstance(ref, (bytes, bytearray)):          # ref images must satisfy the same file specs
        ref_img = cv2.imdecode(np.frombuffer(bytes(ref), np.uint8), cv2.IMREAD_COLOR)
        if ref_img is not None:
            ref_img, _ = _fit(ref_img, max_long, min_short)
            ref = cv2.imencode(".jpg", ref_img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
    body = {}
    if effect == "hair":                       # exactly one of template_id / ref_file_*
        body["hair_color"] = hair_color
        if template_id: body["template_id"] = template_id
        else:           body.update(_src(client, ref, "ref_file"))
    elif effect == "makeup":
        body.update(_src(client, ref, "ref_file"))          # ref is required for mu-transfer
    elif effect == "smile":
        body["expression_type"] = expression_type or SMILE_TYPE
    elif effect == "skin":
        body.update({k: float(level) for k in SKIN_CONCERNS})
    body.update(_src(client, src, "src_file"))
    gen = client.run_generative(BASE + path, body, age=age_target if effect == "aging" else None)
    return _composite(orig, gen, rect) if orig is not None else gen

def _read(src):
    """Accept a local path, bytes, or a public URL for `src`."""
    if isinstance(src, str) and not src.startswith("http") and os.path.exists(src):
        with open(src, "rb") as fh: return fh.read()
    return src
