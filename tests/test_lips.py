"""Lip-colour regression test: the erosion must follow the LIP size, not the image size.

The bug: `r = 0.02 * max(img.shape)` erased the whole lip mask on YouCam's 512-1255px outputs
(lips are ~50x20px), so the lip colour was silently not measured - or, when a few pixels survived,
was compared against an even tighter mask in the reference and scored as fake drift."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from faceprint.attributes import _lip_lab, measure

LIP_BGR, TEETH_BGR = (40, 40, 190), (235, 235, 235)


def face(size, lip_h=20, with_teeth=False, bg=(60, 60, 60), full=True):
    """Synthetic canvas with one lip rectangle and (optionally) a full 106-point landmark set."""
    img = np.full((size, size, 3), bg, np.uint8)
    x0, y0, x1, y1 = size // 2 - 60, size // 2, size // 2 + 60, size // 2 + lip_h
    cv2.rectangle(img, (x0, y0), (x1, y1), LIP_BGR, -1)
    if with_teeth:
        ix = int(0.3 * (x1 - x0)); iy = int(0.30 * lip_h)
        cv2.rectangle(img, (x0 + ix, y0 + iy), (x1 - ix, y1 - iy), TEETH_BGR, -1)
    lm = np.zeros((106, 2))
    lm[52:72] = ([(x0 + i * (x1 - x0) / 9, y0) for i in range(10)] +
                 [(x1 - i * (x1 - x0) / 9, y1) for i in range(10)])
    if full:                       # enough of a face for _aligned()'s eye/nose reference points
        lm[33:43] = [(size / 2 - 90 + i * 6, size / 2 - 60) for i in range(10)]
        lm[87:97] = [(size / 2 + 50 + i * 6, size / 2 - 60) for i in range(10)]
        lm[72:87] = [(size / 2 - 25 + (i % 5) * 12, size / 2 - 40 + (i // 5) * 14) for i in range(15)]
        lm[0:33] = [(size / 2 - 160 + (i % 11) * 32, size / 2 - 200 + (i // 11) * 100) for i in range(33)]
        lm[43:52] = [(size / 2 - 95 + i * 5, size / 2 - 80) for i in range(9)]
        lm[97:106] = [(size / 2 + 45 + i * 5, size / 2 - 80) for i in range(9)]
    return img, lm


def lab_of(bgr):
    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB).astype(float)[0, 0] * [100 / 255, 1, 1] - [0, 128, 128]


def close(a, b, tol=12):
    return a is not None and all(abs(float(x) - float(y)) < tol for x, y in zip(a, b))


target = lab_of(LIP_BGR)

# 1. measurable on every YouCam output size - this is where the image-proportional erosion died
for size in (512, 1024, 1255):
    img, lm = face(size)
    got = _lip_lab(img, lm)
    assert close(got, target, 16), (size, got, target)
    print(f"OK lips measured at {size}px -> L*{got[0]:.0f} a*{got[1]:+.0f} b*{got[2]:+.0f}")

# 2. an open mouth: the interior (teeth) must not become the measurement
img, lm = face(1024, lip_h=46, with_teeth=True)
got = _lip_lab(img, lm)
assert close(got, target, 20), ("mouth interior leaked into the lip colour", got, target)
print(f"OK open mouth interior excluded -> a*{got[1]:+.0f} (lipstick red, not teeth white)")

# 3. unmeasurable lips are skipped, never scored as black
dark = np.full((512, 512, 3), (2, 2, 2), np.uint8); lm2 = np.zeros((106, 2))
lm2[52:72] = np.array([(200 + i * 6, 300) for i in range(10)] + [(320 - i * 6, 322) for i in range(10)])
assert _lip_lab(dark, lm2) is None, "a dark interior must read as not measurable, not as black"
tiny = np.full((512, 512, 3), LIP_BGR, np.uint8); lm3 = np.zeros((106, 2))
lm3[52:72] = np.array([(240 + i * 4, 300) for i in range(10)] + [(280 - i * 4, 316) for i in range(10)])   # 40x16 px lips
assert _lip_lab(tiny, lm3) is not None, "real lip-sized masks must still be measured"
print("OK tiny lips measured / dark interior -> None (skipped, never a fake black reading)")

# 4. measure() carries it through, and None never becomes zeros
img, lm = face(768)
m = measure(img, lm, None)
assert m["lip_lab"] is not None and not np.allclose(np.asarray(m["lip_lab"]), 0.0)
print("OK measure() exposes lip_lab:", [round(float(v)) for v in m["lip_lab"]])
