"""Talent policy: contractual-style rules checked against every AI image in a campaign.
This is a REVIEW aid, not legal evidence."""
import json
from dataclasses import dataclass, field, asdict
from .qa import thresholds

@dataclass
class Policy:
    talent: str = "Talent"
    campaign: str = "Campaign"
    min_cosine: float | None = None          # None -> use the calibrated SAFE threshold
    locked: dict = field(default_factory=dict)  # {"Face shape": 70, "Eyes": 70}: minimum category preservation score

    @classmethod
    def load(cls, path):
        return cls(**json.load(open(path)))

    def save(self, path):
        json.dump(asdict(self), open(path, "w"), indent=2)

def evaluate(policy, res):
    """Return (verdict, [(severity, message)]) for one check() result."""
    th, v = thresholds(), []
    if res["status"] == "NO FACE":
        return "NON-COMPLIANT", [("violation", "No face detected in the image")]
    need = policy.min_cosine if policy.min_cosine is not None else th["safe"]
    if res["cosine"] < th["review"]:
        v.append(("violation", f"Identity drift: similarity {res['cosine']:.2f} is below the drift boundary {th['review']:.2f}"))
    elif res["cosine"] < need:
        v.append(("review", f"Similarity {res['cosine']:.2f} is below the required {need:.2f}"))
    for cat, minimum in policy.locked.items():
        sc = res["categories"].get(cat)
        if sc is not None and sc < minimum:
            v.append(("violation", f"Locked attribute '{cat}' changed too much ({sc:.0f} < {minimum:g})"))
    verdict = "NON-COMPLIANT" if any(s == "violation" for s, _ in v) else "REVIEW" if v else "COMPLIANT"
    return verdict, v
