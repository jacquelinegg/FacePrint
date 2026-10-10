"""Campaign Identity QA report (PDF) with a tamper-evident report id (SHA-256 over images + results)."""
import hashlib, io, os, datetime
import cv2
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from .policy import evaluate
from .qa import thresholds

COL = {"COMPLIANT": colors.HexColor("#C98A57"), "REVIEW": colors.HexColor("#B9A99C"), "NON-COMPLIANT": colors.HexColor("#F1EAE3")}

def _font():
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/Library/Fonts/Arial Unicode.ttf",
              "C:/Windows/Fonts/arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        if os.path.exists(p):
            pdfmetrics.registerFont(TTFont("Body", p)); return "Body"
    return "Helvetica"

def _thumb(img, w_mm):
    h, w = img.shape[:2]; k = 480 / max(h, w)
    small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
    buf = io.BytesIO(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
    return Image(buf, width=w_mm * mm, height=w_mm * mm * small.shape[0] / small.shape[1])

def build(path, policy, ref_img, items):
    """items: [{'name', 'img', 'result'(Baseline.check output), 'yc_changes'(optional list)}]"""
    font, th = _font(), thresholds()
    ss = getSampleStyleSheet()
    P = lambda n, **k: ParagraphStyle(n, parent=ss["Normal"], fontName=font, **k)
    body, small, h1, h2 = P("b", fontSize=9, leading=12), P("s", fontSize=7.5, leading=10, textColor=colors.grey), \
        P("h1", fontSize=20, leading=24, spaceAfter=4), P("h2", fontSize=12, leading=15, spaceBefore=6, spaceAfter=3)
    rows = []
    for it in items:
        verdict, viol = evaluate(policy, it["result"]); rows.append((it, verdict, viol))
    h = hashlib.sha256()
    h.update(cv2.imencode(".png", ref_img)[1].tobytes())
    for it, verdict, viol in rows:
        h.update(cv2.imencode(".png", it["img"])[1].tobytes()); h.update(f"{it['name']}{it['result']['cosine']:.4f}{verdict}".encode())
    rid = h.hexdigest()[:16].upper()
    counts = {k: sum(1 for r in rows if r[1] == k) for k in COL}

    S = [Paragraph("FACEPRINT — Campaign Identity QA", h1),
         Paragraph(f"<b>Talent:</b> {policy.talent} &nbsp;&nbsp; <b>Campaign:</b> {policy.campaign} &nbsp;&nbsp; "
                   f"<b>Date:</b> {datetime.date.today().isoformat()} &nbsp;&nbsp; <b>Report ID:</b> {rid}", body), Spacer(1, 4 * mm)]
    summ = Table([[_thumb(ref_img, 38), Paragraph(
        f"<b>{len(rows)} images checked</b><br/><font color='#C98A57'>{counts['COMPLIANT']} compliant</font> · "
        f"<font color='#B9A99C'>{counts['REVIEW']} need review</font> · "
        f"<font color='#F1EAE3'>{counts['NON-COMPLIANT']} non-compliant</font><br/><br/>"
        f"Rules: similarity ≥ {policy.min_cosine if policy.min_cosine is not None else th['safe']:.2f}; "
        f"drift boundary {th['review']:.2f}; locked attributes: "
        f"{', '.join(f'{k} ≥ {v}' for k, v in policy.locked.items()) or 'none'}.<br/>"
        f"Thresholds: {th['source']}.", body)]], colWidths=[42 * mm, 130 * mm])
    summ.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")])); S += [summ, Spacer(1, 4 * mm)]
    tbl = [["Image", "File", "Similarity", "Identity status", "Policy verdict"]]
    for it, verdict, viol in rows:
        tbl.append([_thumb(it["img"], 18), Paragraph(it["name"], body), f"{100 * max(it['result']['cosine'], 0):.0f}%",
                    it["result"]["status"], Paragraph(f"<font color='{COL[verdict].hexval()}'><b>{verdict}</b></font>", body)])
    t = Table(tbl, colWidths=[22 * mm, 60 * mm, 25 * mm, 32 * mm, 33 * mm], repeatRows=1)
    t.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), font), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f4efe7")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey)]))
    S += [t, PageBreak()]
    for it, verdict, viol in rows:
        r = it["result"]; cats = r["categories"]
        det = [["Attribute group", "Preservation"]] + [[c, f"{v:.0f}"] for c, v in cats.items()]
        dt = Table(det, colWidths=[28 * mm, 17 * mm]); dt.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), font),
                   ("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey)]))
        txt = [Paragraph(f"{it['name']} — <font color='{COL[verdict].hexval()}'>{verdict}</font>", h2),
               Paragraph(f"Similarity {100 * max(r['cosine'], 0):.0f}% ({r['status']})"
                         + (f" · {r['n_faces']} faces in frame, best match compared" if r.get("n_faces", 1) > 1 else ""), body)]
        for sev, msg in viol: txt.append(Paragraph(f"• {msg}", body))
        for a in r["advice"]: txt.append(Paragraph(f"→ {a}", small))
        for c in it.get("yc_changes") or []:
            txt.append(Paragraph(f"YouCam: {c['category']} / {c['attribute']}: {c['before']} → {c['after']}", small))
        right = [dt] if cats else []
        S.append(KeepTogether([Table([[_thumb(it["img"], 45), txt, right]], colWidths=[50 * mm, 80 * mm, 45 * mm],
                  style=[("VALIGN", (0, 0), (-1, -1), "TOP")]), Spacer(1, 5 * mm)]))
        moved = [c for c in (r.get("changes") or []) if c.get("score") is not None and c["score"] < 80]
        if moved:                                   # full-width measurement breakdown under the block
            det2 = [["What changed (reference -> variant)", "Reference", "Variant", "Change"]]
            for c in moved[:12]:
                det2.append([c["attribute"], str(c["reference"]), str(c["variant"]), str(c["change"])])
            if len(moved) > 12: det2.append([f"+ {len(moved) - 12} more measured changes (see the app)", "", "", ""])
            dt2 = Table(det2, colWidths=[68 * mm, 40 * mm, 40 * mm, 32 * mm], repeatRows=1)
            dt2.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), font), ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                                     ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f4efe7")),
                                     ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#6b5433")),
                                     ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                                     ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                     ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#faf7f1")])]))
            S += [dt2, Spacer(1, 6 * mm)]
    S += [Spacer(1, 4 * mm), Paragraph(
        "Method: ArcFace (InsightFace buffalo_l) cosine similarity between the reference photo and each AI image; "
        "attribute scores combine landmark geometry with region-appearance comparison (heuristics, not calibrated on people). This is a visual-similarity review aid. It is not a biometric "
        "identification result, not an authenticity proof and not legal evidence. Use only with the talent's consent.", small)]
    SimpleDocTemplate(path, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                      title=f"FACEPRINT report {rid}").build(S)
    return {"report_id": rid, "counts": counts, "rows": [(r[0]["name"], r[1]) for r in rows]}
