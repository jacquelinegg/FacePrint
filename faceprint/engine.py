"""Face detection + ArcFace embeddings + 106 landmarks + age/gender (InsightFace buffalo_l)."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Face:
    embedding: np.ndarray   # L2-normalised ArcFace vector (512,)
    bbox: np.ndarray
    lm: np.ndarray          # (106, 2) landmarks in image pixels
    det_score: float
    age: float | None = None
    gender: int | None = None   # 1 = male, 0 = female (InsightFace convention)
    kps: np.ndarray | None = None   # (5, 2) eye/nose/mouth keypoints, image pixels

class FaceEngine:
    _app = None

    def __init__(self):
        if FaceEngine._app is None:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l",
                               allowed_modules=["detection", "recognition", "landmark_2d_106", "genderage"],
                               providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            FaceEngine._app = app
        self.app = FaceEngine._app

    def analyze_all(self, img_bgr):
        """All faces found in the image, largest first."""
        out = []
        for f in self.app.get(img_bgr):
            age = getattr(f, "age", None); gen = getattr(f, "gender", None); kps = getattr(f, "kps", None)
            out.append(Face(f.normed_embedding.astype(np.float32), f.bbox, f.landmark_2d_106, float(f.det_score),
                            None if age is None else float(age), None if gen is None else int(gen),
                            None if kps is None else np.asarray(kps, dtype=np.float32)))
        out.sort(key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return out

    def analyze(self, img_bgr, reference=None):
        """Return (face, n_faces). Without `reference` -> the largest face; with a reference embedding -> the
        face that matches it best (so a second person in the frame cannot hide or fake a match)."""
        faces = self.analyze_all(img_bgr)
        if not faces:
            return None, 0
        if reference is None or len(faces) == 1:
            return faces[0], len(faces)
        return max(faces, key=lambda f: cosine(reference, f.embedding)), len(faces)

def cosine(a, b):
    return float(np.dot(a, b))
