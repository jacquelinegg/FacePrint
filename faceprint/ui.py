"""FACEPRINT visual language - graphite + bronze + bone. No traffic-light hues: status is carried by
glyph and density (filled / half / crossed / hollow), never by red-green-yellow.

One tonal scale, three hairline weights, three radii, one signature shape (the aperture) repeated in
the masthead, the labels and the status chips. Charts use a metallic tonal ramp instead of a
categorical rainbow."""
import streamlit as st

try:                                    # st.html() replaces the deprecated components API in newer builds
    from streamlit.components.v1 import html as _html
except Exception:                      # pragma: no cover
    _html = None

APERTURE = (
    '<svg viewBox="0 0 44 44" width="30" height="30" fill="none" stroke="#C98A57" stroke-width="1.6">'
    '<path d="M22 3 38.5 12.5v19L22 41 5.5 31.5v-19z" opacity=".85"/>'
    '<path d="M22 12 32 17.5v11L22 34 12 28.5v-11z" opacity=".55"/>'
    '<path d="M22 12v22M12 17.5l20 11M32 17.5l-20 11" opacity=".38"/></svg>')

STATUS = {"SAFE": "●", "REVIEW": "◐", "IDENTITY DRIFT": "✕", "NO FACE": "○"}
TAB_GLYPHS = {"Campaign QA": "▤", "Stress Test": "↗", "Identity Lock": "◉",
              "Campaign Report": "▭", "YouCam attributes": "◈", "Calibration": "◎"}
ICON = ("data:image/svg+xml," + __import__("urllib.parse", fromlist=["quote"]).quote(
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 44 44'>"
    "<g fill='none' stroke='#C98A57' stroke-width='3.2' stroke-linejoin='round'>"
    "<path d='M22 3.5 38.8 12.7v18.6L22 40.5 5.2 31.3V12.7z' opacity='.92'/>"
    "<path d='M22 13.5 31 18.2v9.6L22 33l-9-5.2v-9.6z' opacity='.62'/></g></svg>"))

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400;1,600&family=Space+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  /* graphite scale - one warm-brown family, four steps */
  --ink:#0D0B0C; --ink2:#100D0E; --panel:#171113; --panel2:#1C1416; --panel3:#231A1D;
  /* hairlines - same warm bone at three weights */
  --hair:rgba(241,234,227,.13); --hair2:rgba(241,234,227,.07); --hair3:rgba(241,234,227,.26);
  /* type */
  --bone:#F1EAE3; --dim:#A7978C; --faint:#6C6058;
  /* one metallic accent + plum/ruter supporting tones - no green, no traffic-light hues */
  --bronze:#C98A57; --bronze-lo:rgba(201,138,87,.38); --bronze-soft:#E3B58C; --taupe:#B9A99C;
  --plum:#8E6E7A; --amber:#E0A874; --rust:#B98A72;
  /* shapes */
  --r-chip:4px; --r-card:14px 3px 14px 3px; --r-head:18px 3px 0 0;
  --serif:'Playfair Display',Georgia,serif;
  --ui:'Space Grotesk',system-ui,-apple-system,'Segoe UI',sans-serif;
  --mono:'JetBrains Mono',ui-monospace,Consolas,monospace;
}
html,body,[class*="css"]{font-family:var(--ui);color:var(--bone);}
.stApp{background:
  radial-gradient(84% 52% at 78% -6%, rgba(201,138,87,.13), transparent 62%),
  radial-gradient(62% 46% at 2% 0%, rgba(142,110,122,.10), transparent 60%),
  radial-gradient(120% 92% at 50% 122%, rgba(201,138,87,.05), transparent 58%),
  radial-gradient(rgba(241,234,227,.05) 1px, transparent 1.2px),
  url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='g'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='4'/%3E%3C/filter%3E%3Crect width='200' height='200' filter='url(%23g)' opacity='.08'/%3E%3C/svg%3E"),
  var(--ink);
  background-size:auto,auto,auto,7px 7px,200px 200px,auto;background-attachment:fixed;}
/* header stays alive: the "expand sidebar" button lives INSIDE stToolbar, so the toolbar must stay
   visible - only the hamburger, status, deploy and logo are removed from it */
header[data-testid="stHeader"]{background:transparent!important;height:2.4rem;min-height:2.4rem;border:none;z-index:999;}
header[data-testid="stHeader"] [data-testid="stToolbar"]{background:transparent!important;border:none!important;
  box-shadow:none!important;left:0!important;right:auto!important;padding-left:2px!important;}
header [data-testid="stToolbar"] #MainMenu,
header [data-testid="stToolbar"] [data-testid="stStatusWidget"],
header [data-testid="stToolbar"] [data-testid="stLogo"],
header [data-testid="stToolbar"] .stDeployButton{display:none!important}
[data-testid="stExpandSidebarButton"] button{color:var(--bronze)!important;opacity:.9;background:transparent!important;
  border:none!important}
[data-testid="stExpandSidebarButton"] button:hover{opacity:1;transform:scale(1.06)}
/* when the sidebar is collapsed, its expand control is a floating button on the left */
[data-testid="stSidebarCollapsedControl"] button{background:rgba(28,20,22,.94)!important;border:1px solid var(--hair)!important;
  border-radius:var(--r-chip)!important;width:36px!important;height:36px!important;opacity:1!important;
  box-shadow:0 4px 18px rgba(0,0,0,.55)}
[data-testid="stSidebarCollapsedControl"] button svg{color:var(--bronze)!important}
::selection{background:var(--bronze);color:var(--ink);}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:var(--ink2)}
::-webkit-scrollbar-thumb{background:rgba(201,138,87,.3);border-radius:8px}
::-webkit-scrollbar-thumb:hover{background:var(--bronze)}

/* ---------- surfaces ---------- */
[data-testid="stMainBlockContainer"]{position:relative;padding-top:1.6rem;padding-bottom:4.5rem;
  max-width:1180px;padding-right:300px}   /* 300px reserved on the right = the permanent workflow column */
[data-testid="stMainBlockContainer"]::before{content:"";position:absolute;left:2px;top:1rem;bottom:3rem;width:1px;
  background:linear-gradient(180deg,transparent,var(--hair) 8%,var(--hair) 92%,transparent);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,var(--ink2),var(--ink));border-right:1px solid var(--hair2);}
[data-testid="stSidebar"]::after{content:"";position:absolute;top:1rem;right:0;bottom:1rem;width:1px;
  background:linear-gradient(180deg,transparent,var(--bronze-lo),transparent);}
[data-testid="stSidebarUserContent"]{padding-top:26px;}

/* ---------- masthead ---------- */
.fp-brand{position:relative;padding:28px 8px 26px;margin-bottom:6px;border-radius:var(--r-head);
  background:linear-gradient(180deg,rgba(36,30,44,.5),rgba(13,11,12,0) 74%);}
.fp-brand::before{content:"";position:absolute;left:0;right:0;top:0;height:1px;
  background:linear-gradient(90deg,var(--hair3),var(--hair) 32%,transparent);}
.fp-eyebrow{display:flex;align-items:center;gap:12px;font-family:var(--mono);font-size:10px;letter-spacing:.34em;
  text-transform:uppercase;color:var(--dim);}
.fp-eyebrow .dot{width:5px;height:5px;border-radius:50%;background:var(--bronze);box-shadow:0 0 10px rgba(201,138,87,.9);}
.fp-eyebrow .rule{flex:0 0 40px;height:1px;background:var(--hair);}
.fp-mast{display:flex;align-items:flex-end;justify-content:space-between;gap:26px;flex-wrap:wrap;margin-top:10px;}
.fp-mast h1{font-family:var(--serif);font-weight:600;font-size:clamp(40px,6.4vw,80px);line-height:.92;letter-spacing:.012em;
  margin:0;text-transform:uppercase;background:linear-gradient(96deg,var(--bone) 12%,var(--bronze) 52%,var(--bronze-soft) 74%,var(--bone) 96%);
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;color:var(--bronze);}
.fp-mast h1 .dot{color:var(--bronze);-webkit-text-fill-color:var(--bronze);}
.fp-offset{position:relative;}
.fp-offset::before{content:attr(data-t);position:absolute;left:2.5px;top:1.5px;color:var(--plum);opacity:.75;z-index:-1;
  font-family:var(--serif);font-weight:600;font-size:inherit;letter-spacing:inherit;text-transform:uppercase;
  -webkit-text-fill-color:var(--plum);}
.fp-sub{font-family:var(--serif);font-style:italic;font-size:clamp(16px,1.9vw,21px);color:var(--taupe);margin:12px 2px 0;
  max-width:62ch;line-height:1.45;}
.fp-spec{font-family:var(--mono);font-size:10px;letter-spacing:.15em;text-transform:uppercase;color:var(--faint);
  text-align:right;line-height:2.1;border-left:1px solid var(--hair2);padding-left:16px;}
.fp-spec b{color:var(--bronze-soft);font-weight:500}
.fp-sprockets{height:15px;margin:20px 0 2px;border-top:1px solid var(--hair2);border-bottom:1px solid var(--hair2);
  background:repeating-linear-gradient(90deg,transparent 0 14px,rgba(241,234,227,.1) 14px 15px,transparent 15px 24px);
  -webkit-mask-image:radial-gradient(circle 5px at 9px 7.5px,transparent 98%,#000 100%);
  mask-image:radial-gradient(circle 5px at 9px 7.5px,transparent 98%,#000 100%);
  mask-size:24px 100%;mask-repeat:repeat-x;}

/* ---------- right guide panel: flush with the page edge, top on the masthead ---------- */
.fp-rail{position:fixed;right:0;top:62px;width:212px;z-index:998;font-family:var(--mono);
  background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--hair2);
  border-right:none;border-radius:14px 0 0 3px;padding:17px 14px 15px 15px;overflow:hidden}
.fp-rail::before{content:"";position:absolute;inset:0 0 auto 0;height:1px;
  background:linear-gradient(90deg,var(--bronze),transparent 70%)}
.fp-rail::after{content:"";position:absolute;right:-26px;top:-26px;width:70px;height:70px;border-radius:50%;
  background:radial-gradient(circle,rgba(241,234,227,.14) 1px,transparent 1.6px);background-size:5px 5px;opacity:.6}
.fp-rail .head{font-size:10px;font-weight:500;letter-spacing:.24em;text-transform:uppercase;color:var(--bronze);
  display:flex;align-items:center;gap:8px;margin:0 12px 13px 0;position:relative}
.fp-rail .head::before{content:"*";color:var(--plum);font-size:12px}
.fp-rail .head::after{content:"";flex:1;height:1px;background:var(--hair)}
.fp-rail .spine{position:absolute;left:21px;top:50px;bottom:18px;width:1px;
  background:linear-gradient(180deg,transparent,var(--hair) 6%,var(--hair) 94%,transparent)}
.fp-rail .steps{display:flex;flex-direction:column;gap:2px;position:relative}
.fp-rail a{display:flex;align-items:flex-start;gap:9px;padding:6px 4px 7px 0;text-decoration:none!important;
  border-left:1px solid transparent;transition:.18s ease}
.fp-rail .n{font-size:9.5px;letter-spacing:.06em;color:var(--faint);margin-top:3px;width:13px;flex:0 0 13px;
  transition:.18s ease}
.fp-rail .mark{width:5px;height:5px;margin-top:6px;flex:0 0 5px;background:var(--faint);
  transform:rotate(45deg);transition:.18s ease}
.fp-rail .t{font-size:11px;letter-spacing:.11em;text-transform:uppercase;color:var(--dim);line-height:1.45;
  transition:.18s ease}
.fp-rail .why{display:block;font-size:10px;letter-spacing:.01em;text-transform:none;color:var(--dim);
  opacity:.85;margin-top:3px;line-height:1.45;max-width:162px}
.fp-rail a:hover{border-left-color:var(--bronze);background:linear-gradient(90deg,rgba(201,138,87,.1),transparent)}
.fp-rail a:hover .t,.fp-rail a:hover .n{color:var(--bronze-soft)}
.fp-rail a:hover .mark{background:var(--bronze);box-shadow:0 0 8px rgba(201,138,87,.6)}
.fp-rail a:hover .why{opacity:1;color:var(--bone)}
@media (max-width:1200px){.fp-rail{display:none}}

/* ---------- labels, rules, chips ---------- */
.fp-label{font-family:var(--mono);font-size:10px;font-weight:500;letter-spacing:.26em;text-transform:uppercase;
  color:var(--bronze);margin:2px 0 12px;display:flex;align-items:center;gap:9px;}
.fp-label::before{content:"*";color:var(--plum);font-size:11px;}
.fp-label::after{content:"";flex:0 0 30px;height:1px;background:var(--hair);}
.fp-rule{position:relative;height:1px;background:var(--hair2);margin:26px 0 22px;}
.fp-rule::after{content:"";position:absolute;left:50%;top:-2.5px;width:5px;height:5px;background:var(--bronze);
  transform:translateX(-50%) rotate(45deg);}
.fp-quote{font-family:var(--serif);font-style:italic;font-size:16px;color:var(--taupe);border-left:1px solid var(--bronze-lo);
  padding:2px 0 2px 16px;margin:10px 0;line-height:1.5;}
/* status chips: filled / half / crossed / hollow - shape carries the meaning, not hue */
.fp-chip{display:inline-flex;align-items:center;gap:8px;font-family:var(--mono);font-size:10.5px;letter-spacing:.15em;
  text-transform:uppercase;color:var(--bone);border:1px solid var(--hair);border-radius:var(--r-chip);padding:4px 11px;}
.fp-chip .g{font-size:11px;line-height:1}
.fp-chip.safe{border-color:var(--bronze);background:rgba(201,138,87,.14);}
.fp-chip.safe .g{color:var(--bronze)}
.fp-chip.review{border-color:var(--bronze-lo)}
.fp-chip.review .g{color:var(--bronze-soft)}
.fp-chip.drift{border-color:var(--hair3);
  background:repeating-linear-gradient(135deg,transparent 0 5px,rgba(241,234,227,.07) 5px 6px);}
.fp-chip.drift .g{color:var(--bone)}
.fp-chip.face{border-style:dotted;color:var(--dim)}
.fp-chip.face .g{color:var(--faint)}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"]{background:transparent;border-bottom:1px solid var(--hair2);gap:3px;padding:0 0 0 2px;}
.stTabs [data-baseweb="tab"]{background:transparent!important;border:none!important;border-radius:0!important;
  font-family:var(--ui)!important;font-size:11px!important;font-weight:500;letter-spacing:.15em;text-transform:uppercase;
  color:var(--faint)!important;padding:11px 13px!important;position:relative;transition:color .2s;}
.stTabs [data-baseweb="tab"]:hover{color:var(--bone)!important;}
.stTabs [aria-selected="true"]{color:var(--bronze-soft)!important;background:transparent!important;}
.stTabs [aria-selected="true"]::after{content:"";position:absolute;left:10px;right:10px;bottom:-1px;height:1px;background:var(--bronze);}
.stTabs [aria-selected="true"]::before{content:"";position:absolute;left:50%;top:5px;width:3px;height:3px;border-radius:50%;
  background:var(--plum);transform:translateX(-50%);}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none!important}

/* ---------- buttons ---------- */
.stButton>button,[data-testid="stDownloadButton"]>button{background:transparent!important;border:1px solid var(--hair)!important;
border-radius:var(--r-chip)!important;color:var(--bone)!important;font-family:var(--mono)!important;font-weight:500;font-size:10.5px!important;
  letter-spacing:.17em!important;text-transform:uppercase;padding:10px 19px!important;transition:.22s ease;}
.stButton>button:hover,[data-testid="stDownloadButton"]>button:hover{background:var(--bronze)!important;color:var(--ink)!important;
  border-color:var(--bronze)!important;box-shadow:0 6px 22px rgba(201,138,87,.2);}
.stButton>button[kind="primary"]{border-color:var(--bronze-lo)!important;color:var(--bronze-soft)!important;}

/* ---------- inputs ---------- */
[data-baseweb="input"]>div,[data-baseweb="select"]>div:first-child,[data-baseweb="textarea"]>div{
  background:rgba(13,11,12,.72)!important;border:1px solid var(--hair)!important;border-radius:var(--r-chip)!important;
  font-family:var(--mono)!important;font-size:12px!important;color:var(--bone)!important;}
[data-baseweb="input"]>div:focus-within,[data-baseweb="select"]>div:focus-within{border-color:var(--bronze-lo)!important;
  box-shadow:0 0 0 1px rgba(201,138,87,.26)!important}
[data-baseweb="select"] svg,[data-baseweb="input"] svg{color:var(--bronze)!important}
/* uploader: bronze-tinted frame instead of the theme's black hairline */
[data-testid="stFileUploadDropzone"]{
  background:rgba(13,11,12,.72)!important;border:1px dashed rgba(201,138,87,.5)!important;border-radius:var(--r-chip)!important;
  font-family:var(--mono)!important;font-size:12px!important;color:var(--bone)!important;padding:6px 4px!important}
[data-testid="stFileUploadDropzone"]:hover{border-color:var(--bronze)!important;background:rgba(201,138,87,.06)!important}
[data-testid="stFileUploadDropzone"] small,[data-testid="stFileUploadDropzone"] span{color:var(--dim)!important;font-family:var(--mono)!important}
[data-testid="stFileUploadDropzone"] button{background:rgba(201,138,87,.14)!important;border:1px solid rgba(201,138,87,.5)!important;
  border-radius:var(--r-chip)!important;color:var(--bronze-soft)!important;font-family:var(--mono)!important;font-size:10px!important;
  letter-spacing:.16em!important;text-transform:uppercase}
[data-testid="stFileUploadDropzone"] button:hover{background:var(--bronze)!important;color:var(--ink)!important;border-color:var(--bronze)!important}
[data-testid="stFileUploaderFileData"]{border:1px solid var(--hair2)!important;border-radius:var(--r-chip)!important;
  background:rgba(13,11,12,.5)!important}
[data-testid="stFileUploaderFileData"] *{font-family:var(--mono)!important;font-size:11px!important;color:var(--dim)!important}
.stMultiSelect [data-baseweb="tag"]{background:rgba(201,138,87,.12)!important;border:1px solid var(--hair)!important;
  border-radius:var(--r-chip)!important}
[data-testid="stFileUploadDropzone"]:hover{border-color:var(--bronze-lo)!important}
[data-testid="stFileUploadDropzone"] small,[data-testid="stFileUploadDropzone"] span{color:var(--dim)!important;font-family:var(--mono)!important}
.stMultiSelect [data-baseweb="tag"]{background:rgba(201,138,87,.12)!important;border:1px solid var(--hair)!important;
  border-radius:var(--r-chip)!important}
.stMultiSelect [data-baseweb="tag"] span{color:var(--bone)!important;font-family:var(--mono)!important;font-size:11px!important}

/* slider: bronze dot on a hairline */
[data-baseweb="slider"]>div>div{background:var(--hair)!important;height:1px!important;border-radius:0!important}
[data-baseweb="slider"] div[role="slider"]{background:var(--bronze)!important;border:1px solid rgba(13,11,12,.7)!important;
  border-radius:50%!important;width:14px!important;height:14px!important;box-shadow:0 0 12px rgba(201,138,87,.5);}
.stSlider [data-testid="stTickBarMin"],.stSlider [data-testid="stTickBarMax"]{font-family:var(--mono)!important;color:var(--faint)!important}

/* radio / checkbox: round bronze */
[data-testid="stRadio"] label,[data-testid="stCheckbox"] label{font-family:var(--mono)!important;font-size:11.5px!important;
  letter-spacing:.08em;color:var(--bone)!important}
[data-testid="stRadio"] div[role="radio"]{border-color:var(--faint)!important;border-radius:50%!important}
[data-testid="stRadio"] div[role="radio"][aria-checked="true"]{border-color:var(--bronze)!important;background:var(--bronze)!important;
  box-shadow:inset 0 0 0 3px var(--ink);}
[data-testid="stCheckbox"] div[role="checkbox"]{border-radius:50%!important;border-color:var(--faint)!important}
[data-testid="stCheckbox"] div[role="checkbox"][aria-checked="true"]{background:var(--bronze)!important;border-color:var(--bronze)!important;color:var(--ink)!important}

/* ---------- cards ---------- */
[data-testid="stMetric"]{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--hair2);
  padding:16px 18px 14px;position:relative;overflow:hidden;border-radius:var(--r-card);}
[data-testid="stMetric"]::before{content:"";position:absolute;inset:0 0 auto 0;height:1px;
  background:linear-gradient(90deg,var(--bronze),transparent 60%);}
[data-testid="stMetric"]::after{content:"";position:absolute;right:-14px;top:-14px;width:58px;height:58px;border-radius:50%;
  background:radial-gradient(circle,rgba(241,234,227,.16) 1px,transparent 1.6px);background-size:5px 5px;opacity:.5;}
[data-testid="stMetricLabel"]{font-family:var(--mono)!important;font-size:9.5px!important;font-weight:400;letter-spacing:.26em!important;
  text-transform:uppercase;color:var(--faint)!important}
[data-testid="stMetricValue"]{font-family:var(--serif)!important;font-weight:600!important;font-size:32px!important;color:var(--bronze-soft)!important}
[data-testid="stMetric"]>div>div:last-child{font-family:var(--mono)!important;font-size:10px!important;color:var(--dim)!important;
  letter-spacing:.14em!important;text-transform:uppercase}

[data-testid="stExpander"]{background:rgba(23,17,19,.72);border:1px solid var(--hair2)!important;
  border-radius:var(--r-card)!important;margin-bottom:12px;overflow:hidden;}
[data-testid="stExpander"] summary{font-family:var(--serif)!important;font-style:italic;font-size:16px!important;letter-spacing:.01em;
  color:var(--bone)!important;padding:12px 16px!important}
[data-testid="stExpander"] summary:hover{color:var(--bronze-soft)!important}
[data-testid="stExpander"] summary svg{color:var(--plum)!important}
[data-testid="stExpander"] details{border:none}

[data-testid="stImage"]{position:relative;border:1px solid var(--hair2);border-radius:var(--r-chip);padding:6px;
  background:rgba(13,11,12,.8);}
[data-testid="stImage"]::before{content:"";position:absolute;left:0;right:0;top:0;height:6px;
  background:repeating-linear-gradient(90deg,transparent 0 7px,rgba(241,234,227,.14) 7px 8px,transparent 8px 14px);
  -webkit-mask-image:radial-gradient(circle 2.5px at 4px 3px,transparent 98%,#000 100%);mask-size:14px 100%;
  mask-repeat:repeat-x;}
[data-testid="stImage"]::after{content:"";position:absolute;right:0;bottom:0;width:12px;height:12px;
  border-right:1px solid var(--bronze);border-bottom:1px solid var(--bronze);opacity:.85;}
[data-testid="stImage"] img{filter:saturate(1.02) contrast(1.02)}
[data-testid="stCaptionContainer"] p,[data-testid="stCaptionContainer"]{font-family:var(--mono)!important;font-size:10px!important;
  letter-spacing:.1em!important;text-transform:uppercase;color:var(--faint)!important;margin-top:6px}

/* ---------- data, charts ---------- */
.stDataFrame,[data-testid="stTable"]{border:1px solid var(--hair2)!important;border-radius:var(--r-chip)!important;
  background:rgba(13,11,12,.8)!important}
.stDataFrame thead th,.stDataFrame th{background:rgba(29,20,22,.9)!important;color:var(--bronze-soft)!important;font-family:var(--mono)!important;
 font-size:9.5px!important;letter-spacing:.13em!important;text-transform:uppercase;border-bottom:1px solid var(--hair)!important}
.stDataFrame td{color:var(--bone)!important;border-bottom:1px solid var(--hair2)!important;font-family:var(--mono)!important;font-size:11.5px!important}
.stDataFrame tr:hover td{background:rgba(201,138,87,.05)!important}
[data-testid="stVegaLiteChart"]{background:transparent!important}
[data-testid="stVegaLiteChart"] svg{background:transparent!important}
[data-testid="stVegaLiteChart"] text{fill:var(--faint)!important;font-family:var(--mono)!important;font-size:9.5px!important}
[data-testid="stVegaLiteChart"] g.mark-rule>path{stroke:var(--hair2)!important}
[data-testid="stVegaLiteChart"] g.mark-line>path{stroke:var(--hair2)!important}

/* ---------- code, alerts, text ---------- */
.stCode,.stCodeBlock,pre,code{background:rgba(13,11,12,.85)!important;border:1px solid var(--hair2)!important;
  border-left:2px solid var(--bronze-lo)!important;border-radius:var(--r-chip)!important;font-family:var(--mono)!important;
  font-size:12px!important;color:var(--bone)!important}
.stMarkdown p,.stMarkdown li{font-size:15.5px;line-height:1.68;color:#DDD3C6}
.stMarkdown h2,.stMarkdown h3{font-family:var(--serif);font-weight:600;font-size:23px;margin:20px 0 10px;color:var(--bone)}
.stMarkdown code{font-family:var(--mono)!important;background:rgba(201,138,87,.12);color:var(--bronze-soft);padding:1px 5px;
  font-size:12px;border-radius:2px}
/* ---------- alerts: no theme green/red - warm tones only ---------- */
.stSuccess,.stInfo,.stWarning,.stError,[data-baseweb="notification"]{
  background:rgba(29,20,22,.92)!important;border:1px solid var(--hair2)!important;border-left-width:2px!important;
  border-radius:var(--r-chip)!important;font-family:var(--mono)!important;color:var(--bone)!important}
.stSuccess p,.stInfo p,.stWarning p,.stError p{font-family:var(--mono)!important;font-size:12px!important;color:var(--bone)!important}
.stSuccess{border-left-color:var(--bronze)!important}
.stInfo{border-left-color:var(--bronze)!important}
.stWarning{border-left-color:var(--plum)!important}
.stError{border-left-color:var(--rust)!important}
.stSuccess svg,.stInfo svg,.stWarning svg,.stError svg{color:var(--bronze-soft)!important;fill:none!important}

/* ---------- alerts, Streamlit 1.64 markup: stAlert > stAlertContainer > stAlertContent{Info,Success,...} ----------
   info (was blue) and success (was green) become a solid bronze accent block with ink text */
[data-testid="stAlert"],[data-testid="stAlertContainer"],[data-testid="stAlertContentInfo"],
[data-testid="stAlertContentSuccess"],.stInfo,.stSuccess{
  background:var(--bronze)!important;border:1px solid rgba(13,11,12,.22)!important;border-left:2px solid #dab393!important;
  border-radius:var(--r-chip)!important;box-shadow:none!important}
[data-testid="stAlertContentInfo"] p,[data-testid="stAlertContentSuccess"] p,.stInfo p,.stSuccess p,
[data-testid="stAlertContentInfo"],[data-testid="stAlertContentSuccess"]{
  color:var(--ink)!important}
[data-testid="stAlert"] p,[data-testid="stAlertContainer"] p{font-family:var(--mono)!important;font-size:12px!important;
  color:var(--bone)!important;line-height:1.6}
[data-testid="stAlertContentInfo"] svg,[data-testid="stAlertContentSuccess"] svg,.stInfo svg,.stSuccess svg{
  color:var(--ink)!important;fill:none!important}
/* only the deeper layers go transparent - the bronze block above must stay filled */
[data-testid="stAlert"] [data-testid="stMarkdownContainer"],
[data-testid="stAlertContentInfo"] *,[data-testid="stAlertContentSuccess"] *{
  background-color:transparent!important;border-color:rgba(13,11,12,.22)!important}
/* warning/error keep the neutral graphite panel, with their own accent edge */
[data-testid="stAlertContentWarning"]{border-left-color:var(--plum)!important}
[data-testid="stAlertContentError"]{border-left-color:var(--rust)!important}
/* older class-based markup, kept for safety */
.stSuccess,.stInfo,.stWarning,.stError{background:rgba(29,20,22,.92)!important;border:1px solid var(--hair2)!important;
  border-left-width:2px!important;border-radius:var(--r-chip)!important;box-shadow:none!important}
.stSuccess p,.stInfo p,.stWarning p,.stError p{font-family:var(--mono)!important;font-size:12px!important;color:var(--bone)!important}
.stSuccess{border-left-color:var(--bronze)!important}
.stInfo{border-left-color:var(--bronze)!important}
.stWarning{border-left-color:var(--plum)!important}
.stError{border-left-color:var(--rust)!important}

/* metric delta (was green text): a solid bronze strip with ink text - the accent lives in the fill */
[data-testid="stMetricDelta"]{background:var(--bronze)!important;border:none!important;border-radius:2px!important;
  padding:2px 7px!important;display:inline-block!important}
[data-testid="stMetricDelta"],[data-testid="stMetricDelta"] div,[data-testid="stMetricDelta"] p,
[data-testid="stMetricDelta"] span,[data-testid="stMetricDelta"] *{
  color:var(--ink)!important;font-family:var(--mono)!important;font-size:9.5px!important;
  letter-spacing:.14em!important;text-transform:uppercase}
[data-testid="stMetricDelta"] svg{display:none!important}

/* ---------- anything Streamlit tints by itself: notifications, links, tooltips ---------- */
.stSuccess,.stInfo,.stWarning,.stError{background:rgba(29,20,22,.92)!important;border:1px solid var(--hair2)!important;
  border-left-width:2px!important;border-radius:var(--r-chip)!important;box-shadow:none!important}
.stSuccess p,.stInfo p,.stWarning p,.stError p{font-family:var(--mono)!important;font-size:12px!important;color:var(--bone)!important}
.stSuccess{border-left-color:var(--bronze)!important}
.stInfo{border-left-color:var(--bronze)!important}
.stWarning{border-left-color:var(--plum)!important}
.stError{border-left-color:var(--rust)!important}
.stSuccess svg,.stInfo svg,.stWarning svg,.stError svg{color:var(--bronze-soft)!important;fill:none!important}
[data-baseweb="notification"]{background:transparent!important;border:none!important;box-shadow:none!important}
[data-baseweb="notification"] *,[data-baseweb="notification"] *::before,
[data-baseweb="notification"] *::after{background-color:transparent!important;border-color:var(--hair2)!important}

a,.stMarkdown a,.stMarkdown a:link,.stMarkdown a:visited{color:var(--bronze-soft)!important;text-decoration-color:var(--bronze-lo)!important}
a:hover,.stMarkdown a:hover{color:var(--bronze)!important}
[data-testid="stTooltipHoverTarget"] svg,[data-testid="stTooltipIcon"] svg,
.stTooltipIcon svg,[data-testid="stTooltipHoverTarget"]{color:var(--dim)!important;fill:none!important;opacity:.8}
/* widget focus rings and selection chrome: bronze instead of the theme blue */
*:focus-visible{outline:1px solid var(--bronze-lo)!important;outline-offset:1px}
::selection{background:var(--bronze);color:var(--ink)}
/* spinner: serif italic, bronze - the only text Streamlit renders outside our components */
[data-testid="stSpinner"]{font-family:var(--serif)!important;font-style:italic;font-size:15.5px!important;
  letter-spacing:.015em;color:var(--bronze-soft)!important}
[data-testid="stSpinner"] *{font-family:var(--serif)!important;font-style:italic;color:var(--bronze-soft)!important}
.stSpinner>div{border-color:var(--bronze)!important;border-right-color:transparent!important}
[data-testid="stSidebar"] .stMarkdown p,[data-testid="stSidebar"] .stMarkdown li{font-family:var(--mono)!important;font-size:12px!important;color:#C6BAAC}
[data-testid="stSidebar"] .stMarkdown strong{color:var(--bronze-soft)!important;letter-spacing:.1em;font-weight:500}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{color:var(--faint)!important}
"""

_CHIP = {"SAFE": "safe", "REVIEW": "review", "IDENTITY DRIFT": "drift", "NO FACE": "face"}

def inject():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    _reset_sidebar_state()

def _reset_sidebar_state():
    """Streamlit remembers a collapsed sidebar in localStorage and then offers no way back on some
    versions. Drop that key on load so the left panel (identity legend + disclaimers) opens expanded;
    collapsing it manually still works until the next reload."""
    try:
        (st.html if hasattr(st, "html") else _html)(
            '<script>for(const k of Object.keys(localStorage))'
            'if(k.startsWith("stSidebarCollapsed-"))localStorage.removeItem(k)</script>', height=0)
    except Exception:
        pass

def wordmark(sub: str, spec: str = ""):
    """Aperture mark + metallic-gradient serif wordmark + italic sub + spec strip + film sprockets."""
    st.markdown(
        '<div class="fp-brand">'
        '<div class="fp-eyebrow"><span class="dot"></span>identity field notes<span class="rule"></span>est. 2026</div>'
        f'<div class="fp-mast"><div style="display:flex;align-items:flex-end;gap:16px">{APERTURE}'
        '<h1>Face<span style="-webkit-text-fill-color:var(--bronze)">print</span><span class="dot">.</span></h1></div>'
        f'<div class="fp-spec">{spec}</div></div>'
        f'<div class="fp-sub">{sub}</div><div class="fp-sprockets"></div></div>', unsafe_allow_html=True)

def label(text: str):
    st.markdown(f'<div class="fp-label">{text}</div>', unsafe_allow_html=True)

def anchor(id_, label=None):
    """Invisible scroll target for the guide rail; the rail label is announced to screen readers."""
    st.markdown(f'<div id="{id_}" aria-label="{label or id_}" style="height:0;margin:0"></div>',
                unsafe_allow_html=True)

RAIL = [
    ("01", "Reference", "upload the original photo - it becomes the identity baseline", "fp-step-0", 0),
    ("02", "Variant", "generate one with YouCam or drop any AI output here", "fp-step-1", 0),
    ("03", "Verdict", "identity % and exactly which group drifted", "fp-step-2", 0),
    ("04", "Stress", "11 strengths per mutation - where identity breaks", "fp-step-3", 1),
    ("05", "Lock", "the strongest effect that keeps the person", "fp-step-4", 2),
    ("06", "Report", "the talent policy + PDF for the client", "fp-step-5", 3),
    ("07", "YouCam", "named attributes, from the API itself", "fp-step-6", 4),
    ("08", "Calibration", "the thresholds, measured - not guessed", "fp-step-7", 5),
]

def rail():
    """Fixed guide rail in the left margin: click a step to jump to its tab and section.
    The label says what it is, the sub-line says why you would click it."""
    steps = "".join(
        f'<a href="#{anchor}" data-tab="{tab}"><span class="n">{n}</span><span class="mark"></span>'
        f'<span class="t">{t}<span class="why">{why}</span></span></a>'
        for n, t, why, anchor, tab in RAIL)
    st.markdown(f'<div class="fp-rail"><div class="head">workflow</div>'
                f'<div class="spine"></div><div class="steps">{steps}</div></div>',
                unsafe_allow_html=True)
    try:
        (st.html if hasattr(st, "html") else _html)(
            """<script>
            window._fpRail = function(){
              var parent = window.parent.document;
              parent.querySelectorAll('.fp-rail a').forEach(function(a){
                a.addEventListener('click', function(){
                  var i = parseInt(a.getAttribute('data-tab'), 10);
                  var tabs = parent.querySelectorAll('[data-testid="stTab"]');   // Streamlit 1.64 markup
                  var tab = tabs[i];
                  if (tab && tab.getAttribute('aria-selected') !== 'true') {
                    var btn = tab.tagName === 'BUTTON' ? tab : tab.querySelector('button');
                    (btn || tab).click();
                  }
                  var target = a.getAttribute('href').slice(1);
                  [250, 550, 900, 1400].forEach(function(ms){
                    setTimeout(function(){
                      var el = parent.getElementById(target);
                      if (el) el.scrollIntoView({behavior:'smooth', block:'start'});
                    }, ms);
                  });
                });
              });
            };
            if (window.parent.document.readyState !== 'loading') window._fpRail();
            else window.parent.document.addEventListener('DOMContentLoaded', window._fpRail);
            </script>""", height=0)
    except Exception:
        pass

def quote(text: str):
    st.markdown(f'<div class="fp-quote">“{text}”</div>', unsafe_allow_html=True)

def rule():
    st.markdown('<div class="fp-rule"></div>', unsafe_allow_html=True)

def chip(status: str):
    """Status chip - filled / half / hatched / hollow instead of traffic lights."""
    return (f'<span class="fp-chip {_CHIP.get(status, "face")}"><span class="g">{STATUS.get(status, "○")}</span>'
            f'{status}</span>')

def boundary(th):
    """The 'what is what' strip: verdicts, thresholds and their source - always visible in the main
    area, so it never depends on the sidebar being open."""
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;align-items:center;gap:10px 22px;margin:6px 0 2px">'
        + chip("SAFE") + f'<span style="font-family:var(--mono);font-size:10px;letter-spacing:.1em;color:var(--dim)">'
        f'cosine &ge; {th["safe"]:.2f} - recognisably the same person</span>'
        + chip("REVIEW") + f'<span style="font-family:var(--mono);font-size:10px;letter-spacing:.1em;color:var(--dim)">'
        f'&ge; {th["review"]:.2f} - changed, check the attributes</span>'
        + chip("IDENTITY DRIFT") + f'<span style="font-family:var(--mono);font-size:10px;letter-spacing:.1em;color:var(--dim)">'
        f'&lt; {th["review"]:.2f} - a different face</span>'
        + '</div>', unsafe_allow_html=True)

def ramp(n):
    """Metallic tonal ramp for multi-series charts: bronze -> taupe -> bone (no categorical rainbow)."""
    stops = ["#C98A57", "#B9A99C", "#8E6E7A", "#E3B58C", "#F1EAE3", "#6C6058", "#D8B79B", "#A7978C"]
    return [stops[i % len(stops)] for i in range(max(1, n))]

_AX = {"labelColor": "#6C6058", "title": None, "tickColor": "rgba(241,234,227,.13)", "labelFontSize": 10,
       "labelFont": "JetBrains Mono", "tickSize": 4}
_CFG = {"background": "transparent", "font": "JetBrains Mono", "padding": 6,
        "axis": {"gridColor": "rgba(241,234,227,.07)", "domainColor": "rgba(241,234,227,.13)"},
        "view": {"stroke": "rgba(241,234,227,.07)"}, "legend": {"disable": True}}

def vega_line(frame, y_domain=(0.0, 1.0), height=200, fmt=".2f"):
    """Line chart on our own spec: bronze ramp, mono axes, NO built-in legend (ours is HTML below/above)."""
    import pandas as pd
    key = frame.index.name or "x"
    data = frame.reset_index().rename(columns={frame.reset_index().columns[0]: key})   # unnamed -> 'index'
    records = data.melt(id_vars=key, var_name="series", value_name="y").to_dict("records")
    for r in records:                                   # NaN -> None so the gaps stay gaps in Vega
        for k, v in r.items():
            if isinstance(v, float) and v != v: r[k] = None
    spec = {"$schema": "https://vega.github.io/schema/vega-lite/v5.json", "width": "container", "height": height,
            "data": {"values": records},
            "mark": {"type": "line", "strokeWidth": 1.6, "point": {"filled": True, "size": 22}},
            "encoding": {"x": {"field": key, "type": "quantitative", "scale": {"nice": False}, "axis": {**_AX, "format": fmt}},
                         "y": {"field": "y", "type": "quantitative", "scale": {"domain": list(y_domain)}, "axis": {**_AX, "format": ".2f"}},
                         "color": {"field": "series", "type": "nominal",
                                   "scale": {"range": ramp(frame.shape[1])}, "legend": None},
                         "tooltip": [{"field": key, "type": "quantitative", "format": fmt},
                                     {"field": "series", "type": "nominal"},
                                     {"field": "y", "type": "quantitative", "format": ".2f"}]},
            "config": _CFG}
    return st.vega_lite_chart(spec, use_container_width=True)

def vega_bar(pairs, height=200, domain=(0, 100), color=None):
    """Single-series bar chart (category names on x) - no legend needed, our spec."""
    data = [{"cat": k, "v": (None if v is None else float(v))} for k, v in pairs if v is not None]
    spec = {"$schema": "https://vega.github.io/schema/vega-lite/v5.json", "width": "container", "height": height,
            "data": {"values": data},
            "mark": {"type": "bar", "width": {"band": 0.55},
                     "color": color or ramp(1)[0], "cornerRadiusEnd": 2},
            "encoding": {"x": {"field": "cat", "type": "nominal", "axis": {**_AX, "labelAngle": -35}},
                         "y": {"field": "v", "type": "quantitative", "scale": {"domain": list(domain)},
                               "axis": {**_AX, "format": ".0f"}},
                         "tooltip": [{"field": "cat", "type": "nominal"}, {"field": "v", "type": "quantitative", "format": ".1f"}]},
            "config": _CFG}
    return st.vega_lite_chart(spec, use_container_width=True)

def legend(names, key, label="legend", default=True):
    """Our own legend row - always restorable via the checkbox, independent of any chart widget."""
    cols = ramp(len(names))
    st.checkbox(label, value=default, key=key)
    if st.session_state.get(key, default):
        row = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:7px;margin:0 16px 4px 0;font-family:var(--mono);'
            f'font-size:9.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--dim)">'
            f'<span style="width:16px;height:2px;background:{c};display:inline-block"></span>{n}</span>'
            for n, c in zip(names, cols))
        st.markdown(f'<div style="display:flex;flex-wrap:wrap;align-items:center;margin:2px 0 8px">{row}</div>',
                    unsafe_allow_html=True)
    return cols

