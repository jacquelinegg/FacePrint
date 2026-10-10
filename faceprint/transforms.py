"""Controlled local face mutations (strength s in 0..1) used by the Stress Test.
They stand in for what a generative tool (YouCam, Firefly, Midjourney...) might do.
For real campaign QA, skip these and upload the AI-generated images directly."""
import cv2, numpy as np

def _grid(img):
    h, w = img.shape[:2]
    return np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

def _remap(img, mx, my):
    return cv2.remap(img, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)

def _radial(img, c, r, k):
    xs, ys = _grid(img); dx, dy = xs - c[0], ys - c[1]
    wgt = np.exp(-(dx**2 + dy**2) / (2 * (r / 2) ** 2))
    sc = 1 - k * wgt
    return _remap(img, c[0] + dx * sc, c[1] + dy * sc)

def _mask(img, lm):
    pts = np.vstack([lm[0:33], lm[43:52], lm[97:106]]).astype(np.int32)
    m = np.zeros(img.shape[:2], np.float32)
    cv2.fillConvexPoly(m, cv2.convexHull(pts), 1.0)
    k = (max(3, int(0.04 * img.shape[1])) * 2 + 1)
    return cv2.GaussianBlur(m, (k, k), 0)[..., None]

def _iod(lm): return float(np.linalg.norm(lm[33:43].mean(0) - lm[87:97].mean(0)))

def big_eyes(img, lm, s):
    for e in (lm[33:43].mean(0), lm[87:97].mean(0)):
        img = _radial(img, e, _iod(lm) * 0.55, 0.55 * s)
    return img

def fuller_lips(img, lm, s):
    m = lm[52:72]
    return _radial(img, m.mean(0), np.ptp(m[:, 0]) * 0.9, 0.5 * s)

def slim_jaw(img, lm, s):
    xs, ys = _grid(img); cx = lm[:33, 0].mean()
    y0, y1 = lm[72:87, 1].mean(), lm[:33, 1].max()
    ramp = np.clip((ys - y0) / (y1 - y0 + 1e-6), 0, 1)
    return _remap(img, cx + (xs - cx) / (1 - 0.30 * s * ramp), ys)

def smooth_skin(img, lm, s):
    d = max(5, int(0.05 * img.shape[1])); sm = img
    for _ in range(1 + int(3 * s)):
        sm = cv2.bilateralFilter(sm, d, 40 + 60 * s, d)
    a = _mask(img, lm) * s
    return (img * (1 - a) + sm * a).astype(np.uint8)

def tone_shift(img, lm, s):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = cv2.cvtColor(np.clip(lab + np.array([30, 4, 18], np.float32) * s, 0, 255).astype(np.uint8),
                       cv2.COLOR_LAB2BGR)
    a = _mask(img, lm)
    return (img * (1 - a) + out * a).astype(np.uint8)

def stylize(img, lm, s):
    return cv2.addWeighted(img, 1 - s, cv2.stylization(img, sigma_s=60, sigma_r=0.45), s, 0)

def smile(img, lm, s):
    """Raise and widen the mouth corners (crude 2-D smile; no teeth are synthesised)."""
    m = lm[52:72]; xs, ys = _grid(img); w = np.ptp(m[:, 0]); r = 0.35 * w
    mx, my = xs.copy(), ys.copy()
    for c, sign in ((m[m[:, 0].argmin()], -1), (m[m[:, 0].argmax()], 1)):
        g = np.exp(-((xs - c[0]) ** 2 + (ys - c[1]) ** 2) / (2 * (r / 1.5) ** 2))
        my += 0.30 * w * s * g          # sample from below -> content moves up
        mx -= sign * 0.12 * w * s * g   # sample from inside -> content moves outwards
    return _remap(img, mx, my)

def head_turn(img, lm, s):
    """Pseudo yaw: slide the inner face sideways more than its contour (2-D approximation of a ~15-20 deg turn)."""
    xs, ys = _grid(img); n = lm[72:87].mean(0); w = np.ptp(lm[:33, 0])
    g = np.exp(-((xs - n[0]) ** 2 + (ys - n[1]) ** 2) / (2 * (0.38 * w) ** 2))
    return _remap(img, xs - 0.16 * w * s * g, ys)

def lipstick(img, lm, s):
    """Makeup-style lip colour shift (towards deep red) restricted to the lip polygon."""
    m = np.zeros(img.shape[:2], np.float32)
    cv2.fillConvexPoly(m, cv2.convexHull(lm[52:72].astype(np.int32)), 1.0)
    k = max(3, int(0.006 * img.shape[1])) * 2 + 1; a = cv2.GaussianBlur(m, (k, k), 0)[..., None] * s
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = cv2.cvtColor(np.clip(lab + np.array([-25, 35, 12], np.float32), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return (img * (1 - a) + out * a).astype(np.uint8)

def beauty_max(img, lm, s):
    for fn in (smooth_skin, tone_shift, big_eyes, fuller_lips, slim_jaw):
        img = fn(img, lm, s)
    return img

TRANSFORMS = {"Skin smoothing": smooth_skin, "Tone / lighting": tone_shift,
              "Bigger eyes": big_eyes, "Fuller lips": fuller_lips,
              "Slimmer jaw": slim_jaw, "Smile": smile, "Head turn": head_turn, "Lipstick": lipstick,
              "Stylize": stylize, "Combined beauty": beauty_max}

def apply(name, img, lm, s):
    return TRANSFORMS[name](img.copy(), lm, float(s))
