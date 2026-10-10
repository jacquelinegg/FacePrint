import json, os, time
import cv2, numpy as np, pandas as pd, streamlit as st
from faceprint.engine import FaceEngine
from faceprint.qa import Baseline, stress_sweep, thresholds, CAL_PATH
from faceprint import transforms as T
from faceprint import ui
from faceprint import attributes

st.set_page_config(page_title="FACEPRINT", page_icon=ui.ICON, layout="wide")
GLYPH = ui.STATUS
ui.rail()
FORMATS = ["jpg", "jpeg", "jpe", "jfif", "pjpeg", "pjp", "png"]   # .jpe/.jfif are JPEG under other names
ui.inject()

@st.cache_resource
def engine(): return FaceEngine()

def read(f):
    return cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)

def rgb(im): return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

th = thresholds()
ui.wordmark("AI identity QA for advertising — how much of a real person survives AI generation?",
            f"engine <b>ArcFace buffalo_l</b><br>thresholds <b>{th['safe']:.2f} / {th['review']:.2f}</b>"
            f"<br>attributes <b>{len(attributes.CATEGORIES)} groups</b>")
st.sidebar.markdown(
    '<div style="font-family:var(--mono);font-size:10px;line-height:2.3;letter-spacing:.05em;color:var(--dim)">'
    + ui.chip("SAFE") + f' cosine &ge; {th["safe"]:.2f} - recognisably the same person<br>'
    + ui.chip("REVIEW") + f' &ge; {th["review"]:.2f} - changed: check the attributes<br>'
    + ui.chip("IDENTITY DRIFT") + f' &lt; {th["review"]:.2f} - a different face<br>'
    + ui.chip("NO FACE") + ' no face detected in the image'
    + '</div>', unsafe_allow_html=True)
st.sidebar.caption(f"Cosine similarity of ArcFace embeddings. Thresholds: {th['source']}.")
st.sidebar.caption("Visual-similarity QA tool. Not a biometric identification result and not legal evidence. Use only with the model's consent.")

ui.anchor("fp-step-0", "Reference")
ref_file = st.file_uploader("Reference photo (original model)", type=FORMATS)
if not ref_file:
    st.info("Upload a reference photo, or start from the bundled demo face - then generate an AI variant "
            "of it in the Campaign QA tab and see how much of the person survives.")
    if st.button("Use demo face", type="primary"): st.session_state["demo"] = True
    if not st.session_state.get("demo"): st.stop()
    for p in ("demo_data/original.jpg", "samples/demo_face.jpg"):
        if os.path.exists(p):
            ref = cv2.imread(p); demo_name = os.path.basename(p); break
    else:
        st.error("No demo image found."); st.stop()
    st.caption(f"Demo reference: `{demo_name}` — generate a YouCam variant in Campaign QA, or upload any AI output.")
else:
    ref = read(ref_file)
try:
    base = Baseline(engine(), ref)
except ValueError as e:
    st.error(str(e)); st.stop()
if base.n_ref_faces > 1:
    st.warning(f"The reference photo contains {base.n_ref_faces} faces - the largest one is used as the Identity Lock. "
               "Crop the photo to the model for a reliable baseline.")
with st.container():
    ui.label("Identity Lock · reference")
    pc1, pc2, pc3 = st.columns([1, 2, 2])
    pc1.image(rgb(ref), caption=f"{ref_file.name if ref_file else 'demo: ' + demo_name}")
    f = base.face
    pc2.metric("Faces in frame", base.n_ref_faces,
               "single person - good baseline" if base.n_ref_faces == 1 else "crop to one person")
    pc2.metric("Face coverage", f"{100 * (f.bbox[2] - f.bbox[0]) / ref.shape[1]:.0f}% of width",
               "auto-cropped to the face before YouCam calls" if (f.bbox[2] - f.bbox[0]) / ref.shape[1] < 0.6
               else "full frame is fine for YouCam")
    pc3.metric("Apparent age (model)", f"{f.age:.0f}" if f.age is not None else "-", "information only")
    pc3.metric("Predicted sex (model)", "male" if f.gender else "female", "information only")
ui.rule()


tab_qa, tab_stress, tab_lock, tab_rep, tab_yc, tab_cal = st.tabs(
    [f"{ui.TAB_GLYPHS[t]} {t}" for t in ("Campaign QA", "Stress Test", "Identity Lock",
                                         "Campaign Report", "YouCam attributes", "Calibration")])

with tab_qa:
    ui.anchor("fp-step-1", "Variant")
    ui.label("Where the AI images come from")
    st.write("Generate a variant with our own YouCam effects, or upload AI output from any generator "
             "(Firefly, Midjourney, ...). Each image is checked against the Identity Lock.")
    with st.expander("Make an AI variant from this reference (one-click demo)", expanded=True):
        c1, c2, c3 = st.columns(3)
        src_mode = c1.radio("Source", ["YouCam effect (real AI)", "local simulation"], horizontal=False)
        if src_mode.startswith("YouCam"):
            eff = c1.selectbox("Effect", ["aging", "hair", "smile", "skin"])
        else:
            eff = None
            mut = c1.selectbox("Mutation", list(T.TRANSFORMS))
            strength = c2.slider("Strength", 0.0, 1.0, 0.7, 0.05)
        age_target = 25
        if src_mode.startswith("YouCam") and eff == "aging":
            age_target = c2.slider("Target age (YouCam res_age series)", 12, 70, 25)
        param = c2.text_input("hair template_id", "female_blunt_bob") if (src_mode.startswith("YouCam") and eff == "hair") else None
        with c3:
            st.write("")
            go = st.button("Generate + add to the analysis", use_container_width=True)
        if go:
            st.session_state.setdefault("gen_variants", [])
            try:
                with st.spinner("YouCam is generating"):
                    if src_mode.startswith("YouCam"):
                        if not os.environ.get("YOUCAM_API_KEY"):
                            st.warning("Set YOUCAM_API_KEY first (or use the local simulation).")
                        else:
                            from faceprint.generators import generate
                            from faceprint.youcam import YouCamClient
                            cl = YouCamClient(strictness=os.environ.get("YOUCAM_STRICTNESS", "low"))
                            name = f"youcam_{eff}"
                            if eff == "aging": name += f"_age{age_target}"
                            if param: name += f"_{param}"
                            t0 = time.time()
                            im = generate(cl, eff, ref, template_id=param if eff == "hair" else None,
                                          age_target=age_target if eff == "aging" else None)
                            st.session_state["gen_variants"].append((name + ".jpg", im))
                            st.caption(f"Generated with YouCam `{eff}` in {time.time()-t0:.0f}s" +
                                       (f" (res_age {cl.last_age:.0f})" if getattr(cl, "last_age", None) else ""))
                            if eff == "aging":
                                f_new, _ = engine().analyze(im)
                                if f_new is not None and f_new.age is not None and base.face.age is not None:
                                    d = f_new.age - base.face.age
                                    st.caption(f"apparent age: {base.face.age:.0f} → {f_new.age:.0f} ({d:+.0f} y)"
                                               + ("" if abs(d) >= 5 else
                                                  " · mid-range aging is subtle on a smooth young face - "
                                                    "try 65+ for a visible change"))
                    else:
                        im = T.apply(mut, ref, base.face.lm, strength)
                        st.session_state["gen_variants"].append((f"local_{mut.replace(' ', '_')}_{strength:.2f}.jpg", im))
                    st.session_state["gen_variants"] = st.session_state["gen_variants"][-6:]
            except Exception as e:
                st.error(f"Generation failed: {e}")
        if st.session_state.get("gen_variants"):
            if st.button("Clear generated variants"):
                st.session_state["gen_variants"] = []
    files = st.file_uploader("…or upload AI-generated variants", type=FORMATS, accept_multiple_files=True)
    items = [(f.name, read(f)) for f in files] + list(st.session_state.get("gen_variants", []))
    rows = []
    for name, im in items:
        r = base.check(im)
        if r["cosine"] >= 0.999:                      # the image IS the reference (uploaded it, or a no-op generator)
            ui.quote("This image is bit-identical to the reference - it says nothing about identity. "
                     "Upload an AI-modified image, or generate one with the button above.")
        rows.append({"Image": name, "Status": f"{GLYPH[r['status']]} {r['status']}", "Identity": round(100 * max(r["cosine"], 0)),
                     **{k: round(v) for k, v in r["categories"].items()}})
        with st.expander(f"{GLYPH[r['status']]} {name} — identity {100*max(r['cosine'],0):.0f}%", expanded=len(items) <= 3):
            c1, c2 = st.columns([1, 2]); c1.image(rgb(im))
            c2.markdown(f"**Preserved:** {', '.join(r['preserved']) or '—'}  \n**Changed:** {', '.join(r['changed']) or '—'}  \n**Lost:** {', '.join(r['lost']) or '—'}")
            if r["n_faces"] > 1: c2.info(f"{r['n_faces']} faces in this image - the one most similar to the reference was compared.")
            if r["categories"]:
                ui.vega_bar(sorted(r["categories"].items()), height=170, domain=(0, 100))
            for a in r["advice"]: ui.quote(a)
            if r["changes"]:
                with c2.expander("Full measurement breakdown (reference -> variant, worst first)", expanded=True):
                    st.dataframe(pd.DataFrame(r["changes"]), use_container_width=True, hide_index=True)
                    st.caption("Colour rows are CIELAB: **L\\*** = lightness (0 black - 100 white), "
                               "**a\\*** = red-green, **b\\*** = yellow-blue, **dE** = perceptual colour "
                               "difference (anything above ~2 is visible to the eye). Ratios like eye width "
                               "are normalised by the inter-ocular distance, so they are scale independent; "
                               "NCC 1.00 = identical pixels.")
    ui.anchor("fp-step-2", "Verdict")
    if rows:
        df = pd.DataFrame(rows); st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV report", df.to_csv(index=False), "faceprint_qa.csv")

with tab_stress:
    ui.anchor("fp-step-3", "Stress")
    st.write("Progressively mutate the reference photo and find where identity leaves the safe zone.")
    st.write("Progressively mutate the reference photo and find where identity leaves the safe zone.")
    st.caption("Two independent layers are measured at every strength: **identity** (ArcFace cosine - stays high under "
               "cosmetic edits because the person is still recognisable) and **attributes** (the interpretable scores - "
               "show what actually changed). A mutation that never drops identity below SAFE but wrecks the attributes "
               "is exactly the case a cosine-only tool misses.")
    chosen = st.multiselect("Mutations", list(T.TRANSFORMS), default=list(T.TRANSFORMS))
    st.caption("11 strengths per mutation (0.0 - 1.0) - the sweep takes about a minute.")
    if st.button("Run stress test") and chosen:
        with st.spinner("Running mutations..."):
            rows, imgs = stress_sweep(base, ref, transforms=chosen)
        df = pd.DataFrame(rows)
        names = list(chosen)
        c1, c2 = st.columns(2)
        with c1:
            ui.label("Identity layer · cosine")
            ui.legend(names, key="leg-identity")
            ui.vega_line(df.pivot(index="strength", columns="transform", values="cosine"), (0.0, 1.0))
        with c2:
            ui.label("Attribute layer · worst group")
            ui.legend(names, key="leg-attribute")
            ui.vega_line(df.pivot(index="strength", columns="transform", values="worst_attribute_score"), (0, 100), fmt=".1f")
        ui.rule()
        last = df[df.review_at.notna() | df.drift_at.notna() | df.attr_at.notna()
                  | (df.strength == df.strength.max())].groupby("transform").tail(1)
        st.dataframe(last[["transform", "review_at", "drift_at", "attr_at", "worst_attribute",
                           "worst_attribute_score"]].rename(columns={
            "review_at": "identity leaves SAFE at", "drift_at": "identity DRIFT / no face at",
            "attr_at": "attributes degraded at", "worst_attribute": "worst attribute",
            "worst_attribute_score": "worst score"}), use_container_width=True)
        steps = sorted({k[1] for k in imgs})
        for name in chosen:
            st.markdown(f"**{name}**"); cols = st.columns(len(steps))
            for col, s in zip(cols, steps):
                c = df[(df["transform"] == name) & (df["strength"] == s)]
                if c.empty: continue
                c = c.iloc[0]
                cos = "no face" if pd.isna(c["cosine"]) else f"{c['cosine']:.2f}"
                worst = f"{c['worst_attribute']} {c['worst_attribute_score']:.0f}" if c["worst_attribute"] else "-"
                col.image(rgb(imgs[(name, s)]), caption=f"{s:.2f} | {cos} {GLYPH[c['status']]} | {worst}")

with tab_lock:
    ui.anchor("fp-step-4", "Lock")
    ui.label("Generative effect + identity lock")
    st.write("Generate a YouCam effect, then find the **strongest version that keeps the person recognisable**.")
    lkey = st.text_input("YouCam API key ", value=os.environ.get("YOUCAM_API_KEY", ""), type="password")
    effect = st.selectbox("Effect", ["aging", "hair", "makeup", "smile", "skin"])
    from faceprint.generators import DEFAULT_TEMPLATE
    c1, c2 = st.columns(2)
    tmpl = c1.text_input("Hair template_id", DEFAULT_TEMPLATE) if effect == "hair" else None
    if effect == "hair" and lkey:
        with c2.expander("Browse hair templates"):
            if st.button("List YouCam templates"):
                from faceprint.youcam import YouCamClient
                st.session_state["tpls"] = YouCamClient(lkey).templates()
            for t in st.session_state.get("tpls", []):
                st.write(f"`{t['id']}` — {t.get('title')} ({t.get('category_name')})")
    ref_look = c2.text_input("Makeup reference look (public URL)") if effect == "makeup" else None
    if effect == "makeup":
        rf = st.file_uploader("…or drop the look image here", type=FORMATS, key="ref_look_file")
        if rf is not None: ref_look = read(rf)
        if not ref_look: st.caption("Give a public URL or a local image - mu-transfer needs a reference look.")
    src_url = st.text_input("Optional: public URL of the original (verified request shape; otherwise the file is uploaded)")
    age_target = None
    if effect == "aging":
        st.caption("Aging returns a whole `res_age` series (12-70) in one call - pick the target age directly.")
        direction = st.radio("Direction", ["Older (strongest)", "Younger (strongest)", "Specific age"], horizontal=True)
        if direction == "Younger (strongest)": age_target = 12
        elif direction == "Specific age": age_target = st.slider("Target age", 12, 70, 30)
    if st.button("Generate + find safe intensity") and lkey:
        from faceprint.youcam import YouCamClient, YouCamError
        from faceprint.generators import generate
        from faceprint.lock import find_safe_intensity, find_safe_native
        from faceprint.generators import NATIVE
        try:
            with st.spinner("YouCam is generating"):
                cl, src = YouCamClient(lkey), src_url or cv2.imencode(".jpg", ref)[1].tobytes()
                if effect in NATIVE:   # native API strength: one call per level
                    gen = generate(cl, effect, src, level=1.0)
                    r = find_safe_native(base, ref, lambda lv: gen if lv == 1.0 else generate(cl, effect, src, level=lv))
                else:
                    gen = generate(cl, effect, src, ref=ref_look, template_id=tmpl, age_target=age_target)
                    r = find_safe_intensity(base, ref, gen)
            full = r["full_effect"]
            if getattr(cl, "last_age", None) is not None:
                st.caption(f"YouCam produced the `res_age` = {cl.last_age:.0f} variant of the aging series.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Full effect", f"{100*max(full['cosine'],0):.0f}%", full["status"])
            m2.metric("Max SAFE intensity", f"{r['safe_intensity']:.0%}")
            m3.metric("Max REVIEW intensity", f"{r['review_intensity']:.0%}")
            i1, i2, i3 = st.columns(3)
            i1.image(rgb(ref), caption="Original"); i2.image(rgb(gen), caption="Full AI effect")
            i3.image(rgb(r["safe_image"]), caption=f"Identity-locked ({r['safe_intensity']:.0%})")
            if effect == "aging":
                from faceprint.youcam import _age_series
                series = _age_series(getattr(cl, "last_results", None)) or []
                if series:
                    ui.label(f"Aging series — {len(series)} variants came back in ONE call (identity measured on each)")
                    st.caption("`res_age` is YouCam's own label for each generated age; the % is the local "
                               "ArcFace identity against the reference. The middle of the range is deliberately "
                               "subtle - the visible drift lives at the extremes.")
                    cols = st.columns(min(8, len(series)))
                    for k, (age, url) in enumerate(series):
                        im = cl.image(url); rr = base.check(im)
                        col = cols[k % len(cols)]
                        col.image(rgb(im), caption=f"{int(age)} · {GLYPH[rr['status']]} {100*max(rr['cosine'],0):.0f}%")
            ui.vega_line(pd.DataFrame(r["curve"]).set_index("intensity")[["cosine"]], (0.0, 1.0), height=180)
        except YouCamError as e:
            st.error(str(e))

with tab_rep:
    ui.anchor("fp-step-5", "Evidence")
    st.write("Check the images uploaded in **Campaign QA** against the talent's rules and download a PDF report.")
    from faceprint.policy import Policy, evaluate
    from faceprint.attributes import CATEGORIES
    c1, c2 = st.columns(2)
    talent = c1.text_input("Talent", "Talent"); camp = c2.text_input("Campaign", "Campaign")
    minc = st.slider("Minimum similarity", 0.0, 1.0, float(th["safe"]), 0.01)
    locked_cats = st.multiselect("Locked attribute groups (must stay preserved)", CATEGORIES)
    locked_min = st.slider("Minimum preservation for locked groups", 40, 95, 70) if locked_cats else 70
    pol = Policy(talent=talent, campaign=camp, min_cosine=minc, locked={c: locked_min for c in locked_cats})
    if not files: st.info("Upload AI variants in the Campaign QA tab first.")
    else:
        from faceprint.report import build
        import tempfile
        items = [{"name": f.name, "img": read(f)} for f in files]
        for it in items: it["result"] = base.check(it["img"])
        for it in items:
            v, viol = evaluate(pol, it["result"]); st.write(f"**{it['name']}** — {v}" + "".join(f"  \n• {m}" for _, m in viol))
        if st.button("Build PDF report"):
            with tempfile.TemporaryDirectory() as d:
                info = build(os.path.join(d, "r.pdf"), pol, ref, items)
                st.session_state["pdf"] = (open(os.path.join(d, "r.pdf"), "rb").read(), info["report_id"])
        if "pdf" in st.session_state:
            st.download_button(f"Download report {st.session_state['pdf'][1]}", st.session_state["pdf"][0], "faceprint_report.pdf", "application/pdf")

with tab_yc:
    ui.anchor("fp-step-6", "YouCam")
    ui.label("YouCam face attribute analysis")
    st.write("Compare the **named** attributes YouCam sees (face shape, eye shape, nose width, ...) between the reference photo "
             "and AI variants. Images are uploaded to YouCam via its File API (kept ~24h). Frontal, single face, face > 60% of the frame.")
    key = st.text_input("YouCam API key", value=os.environ.get("YOUCAM_API_KEY", ""), type="password")
    yc_files = st.file_uploader("AI variants to analyze", type=FORMATS, accept_multiple_files=True, key="yc")
    strict = st.selectbox("Face-angle strictness", ["flexible", "low", "medium", "high", "strict"], index=1,
                          help="Real-world and AI shots are rarely perfectly level - 'low' accepts a head tilt, "
                               "'strict' only textbook-frontal faces. The bundled demo face needs 'low'.")
    if st.button("Analyze with YouCam") and key and yc_files:
        from faceprint.youcam import YouCamClient, YouCamError, compare_results
        try:
            cl = YouCamClient(key, strictness=strict)
            ref_res = cl.analyze_bytes(cv2.imencode(".jpg", ref)[1].tobytes())
            scores, changes = [], []
            for f in yc_files:
                r = compare_results(ref_res, cl.analyze_bytes(f.getvalue())); st.markdown(f"**{f.name}**")
                ui.vega_bar(sorted(r["categories"].items()), height=170, domain=(0, 100))
                if r["changes"]: st.dataframe(pd.DataFrame(r["changes"]), use_container_width=True)
                else: st.success("No YouCam attribute changed")
                scores.append({"Image": f.name, **{k: round(v) for k, v in r["categories"].items()}})
                changes += [{"Image": f.name, **c} for c in r["changes"]]
            c1, c2 = st.columns(2)
            c1.download_button("Attribute scores (CSV)", pd.DataFrame(scores).to_csv(index=False),
                               "youcam_attribute_scores.csv")
            c2.download_button("Attribute changes (CSV)", pd.DataFrame(changes).to_csv(index=False) if changes else "",
                               "youcam_attribute_changes.csv", disabled=not changes)
        except YouCamError as e:
            st.error(str(e))

with tab_cal:
    ui.anchor("fp-step-7", "Calibration")
    st.write(f"Current thresholds: **safe ≥ {th['safe']:.2f}**, **review ≥ {th['review']:.2f}** — _{th['source']}_")
    if os.path.exists(CAL_PATH):
        cal = json.load(open(CAL_PATH)); idx = [f"{b:.2f}" for b in cal["bins"][:-1]]
        piv = pd.DataFrame({"same person": cal["genuine_hist"], "different people": cal["impostor_hist"]},
                           index=[float(b) for b in cal["bins"][:-1]])
        ui.legend(["same person", "different people"], key="leg-calibration")
        ui.vega_line(piv, (0.0, float(max(piv.values.max(), 1))), height=190, fmt=".2f")
    else:
        st.code("python -m faceprint calibrate path/to/dataset   # dataset/<person>/*.jpg (e.g. LFW)")

