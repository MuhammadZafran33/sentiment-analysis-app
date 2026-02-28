import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob
import re
import time

st.set_page_config(
    page_title="SentimentAI — Text Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background: #060609 !important;
    color: #e2e2ef !important;
}
.block-container { padding: 2rem 3rem !important; max-width: 1300px !important; }
#MainMenu, footer, header, .stDeployButton { visibility: hidden !important; display: none !important; }

/* BG Orbs */
.orb { position: fixed; border-radius: 50%; filter: blur(100px); pointer-events: none; z-index: 0; }
.orb1 { width: 600px; height: 600px; background: rgba(99,102,241,0.12); top: -200px; left: -200px; animation: drift 12s ease-in-out infinite; }
.orb2 { width: 500px; height: 500px; background: rgba(236,72,153,0.09); bottom: -100px; right: -100px; animation: drift 10s ease-in-out infinite reverse; }
.orb3 { width: 400px; height: 400px; background: rgba(16,185,129,0.08); top: 40%; left: 30%; animation: drift 14s ease-in-out infinite 2s; }
@keyframes drift { 0%,100%{transform:translate(0,0)} 33%{transform:translate(40px,-40px)} 66%{transform:translate(-30px,30px)} }

/* HERO */
.hero { text-align: center; padding: 3.5rem 1rem 2.5rem; position: relative; z-index: 1; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.35);
    color: #a5b4fc; font-size: 11px; font-weight: 700; letter-spacing: 3px; text-transform: uppercase;
    padding: 7px 22px; border-radius: 100px; margin-bottom: 1.8rem;
}
.hero-badge .dot { width: 7px; height: 7px; background: #818cf8; border-radius: 50%; animation: blink 2s infinite; }
@keyframes blink { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.3;transform:scale(0.7)} }

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.8rem; font-weight: 800; line-height: 1.1;
    letter-spacing: -1.5px; color: #f1f1f8; margin-bottom: 1.2rem;
}
.grad { background: linear-gradient(135deg, #818cf8 0%, #ec4899 55%, #10b981 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.hero-sub { font-size: 1.1rem; color: #8585a8; max-width: 480px; margin: 0 auto 2.2rem; line-height: 1.7; }

.pills { display: flex; justify-content: center; flex-wrap: wrap; gap: 12px; margin-bottom: 2.5rem; }
.pill {
    display: flex; align-items: center; gap: 8px;
    background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.25);
    color: #a5b4fc; font-size: 12px; font-weight: 600;
    padding: 8px 18px; border-radius: 100px;
}
.pill-icon { font-size: 15px; }

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important; border-radius: 14px !important;
    padding: 5px !important; border: 1px solid rgba(255,255,255,0.07) !important; gap: 6px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important; color: #555572 !important;
    font-weight: 600 !important; font-size: 14px !important; padding: 12px 28px !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] { background: rgba(99,102,241,0.18) !important; color: #a5b4fc !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* TEXTAREA */
.stTextArea > div > div > textarea {
    background: rgba(15,15,35,0.95) !important; border: 1.5px solid rgba(99,102,241,0.5) !important;
    border-radius: 18px !important; color: #ffffff !important;
    font-size: 16px !important; font-family: 'Inter', sans-serif !important;
    padding: 18px !important; line-height: 1.7 !important; transition: border-color 0.3s !important;
}
.stTextArea > div > div > textarea:focus {
    border-color: rgba(99,102,241,0.7) !important;
    box-shadow: 0 0 0 4px rgba(99,102,241,0.12) !important;
}
.stTextArea > div > div > textarea::placeholder { color: #4a4a6a !important; font-style: italic; }

/* BUTTONS */
.stButton > button {
    border-radius: 12px !important; font-weight: 600 !important;
    font-size: 13px !important; transition: all 0.25s ease !important; outline: none !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important; border: none !important;
    padding: 14px 32px !important; font-size: 15px !important; letter-spacing: 0.3px !important;
    box-shadow: 0 6px 24px rgba(99,102,241,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important; box-shadow: 0 10px 35px rgba(99,102,241,0.5) !important;
}
.stButton > button:not([kind="primary"]) {
    background: rgba(255,255,255,0.04) !important; color: #6b7280 !important;
    border: 1px solid rgba(255,255,255,0.08) !important; padding: 9px 14px !important; font-size: 12px !important;
}
.stButton > button:not([kind="primary"]):hover {
    background: rgba(99,102,241,0.12) !important; color: #a5b4fc !important;
    border-color: rgba(99,102,241,0.35) !important;
}

/* RESULT CARD */
.result-card {
    border-radius: 22px; padding: 2.8rem 2rem; text-align: center;
    position: relative; overflow: hidden;
    animation: popIn 0.45s cubic-bezier(0.34,1.56,0.64,1) both;
}
@keyframes popIn { from{opacity:0;transform:scale(0.75)} to{opacity:1;transform:scale(1)} }
.rc-pos { background: linear-gradient(145deg, #052e16, #064e3b, #065f46); border: 1px solid rgba(16,185,129,0.35); box-shadow: 0 24px 60px rgba(16,185,129,0.18), inset 0 1px 0 rgba(255,255,255,0.05); }
.rc-neg { background: linear-gradient(145deg, #3b0000, #450a0a, #7f1d1d); border: 1px solid rgba(239,68,68,0.35); box-shadow: 0 24px 60px rgba(239,68,68,0.18), inset 0 1px 0 rgba(255,255,255,0.05); }
.rc-neu { background: linear-gradient(145deg, #111118, #1c1c28, #252532); border: 1px solid rgba(148,163,184,0.2); box-shadow: 0 24px 60px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05); }
.rc-emoji { font-size: 80px; line-height: 1; display: block; margin-bottom: 14px; filter: drop-shadow(0 6px 12px rgba(0,0,0,0.4)); }
.rc-label { font-family: 'Space Grotesk', sans-serif; font-size: 2.6rem; font-weight: 800; color: #fff; letter-spacing: -1px; }
.rc-sub { font-size: 13px; color: rgba(255,255,255,0.65); margin-top: 8px; font-weight: 500; }

/* METRIC GLASS */
.mg {
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px; padding: 22px; text-align: center;
    transition: all 0.3s ease; margin-top: 14px;
}
.mg:hover { background: rgba(99,102,241,0.07); border-color: rgba(99,102,241,0.25); transform: translateY(-3px); }
.mg-label { font-size: 11px; font-weight: 800; letter-spacing: 2.5px; text-transform: uppercase; color: #818cf8; margin-bottom: 10px; }
.mg-val { font-family: 'Space Grotesk', sans-serif; font-size: 2.2rem; font-weight: 800; color: #818cf8; line-height: 1; margin-bottom: 6px; }
.mg-desc { font-size: 12px; color: #7c7c9a; font-weight: 500; }

/* CONFIDENCE */
.conf-wrap { background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07); border-radius: 18px; padding: 22px; margin-top: 14px; }
.conf-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
.conf-title { font-size: 11px; font-weight: 800; letter-spacing: 2.5px; text-transform: uppercase; color: #818cf8; }
.conf-num { font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 800; color: #e2e2ef; }
.conf-track { background: rgba(255,255,255,0.05); border-radius: 100px; height: 10px; overflow: hidden; }
.cf-p { background: linear-gradient(90deg, #047857, #10b981, #6ee7b7); height: 100%; border-radius: 100px; }
.cf-n { background: linear-gradient(90deg, #b91c1c, #ef4444, #fca5a5); height: 100%; border-radius: 100px; }
.cf-u { background: linear-gradient(90deg, #44403c, #78716c, #d6d3d1); height: 100%; border-radius: 100px; }

/* KEYWORDS */
.kw-wrap { background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07); border-radius: 18px; padding: 18px; margin-top: 14px; }
.kw-title { font-size: 11px; font-weight: 800; letter-spacing: 2.5px; text-transform: uppercase; color: #818cf8; margin-bottom: 12px; }
.kw-tags { display: flex; flex-wrap: wrap; gap: 8px; }
.kw-p { background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.28); color: #34d399; padding: 4px 14px; border-radius: 100px; font-size: 12px; font-weight: 600; }
.kw-n { background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.28); color: #f87171; padding: 4px 14px; border-radius: 100px; font-size: 12px; font-weight: 600; }

/* EMPTY STATE */
.empty-state { text-align: center; padding: 4rem 2rem; color: #2a2a40; }
.empty-icon { font-size: 64px; filter: grayscale(1) opacity(0.2); margin-bottom: 16px; display: block; }
.empty-text { font-size: 16px; font-weight: 600; color: #6b6b8a; }

/* SECTION */
.sec-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 700; color: #e2e2ef; margin-bottom: 6px; }
.sec-sub { font-size: 14px; color: #7c7c9a; margin-bottom: 1.4rem; font-weight: 500; }
.ex-label { font-size: 11px; font-weight: 800; letter-spacing: 2.5px; text-transform: uppercase; color: #6366f1; margin: 16px 0 10px; }

/* METRICS */
[data-testid="stMetric"] { background: rgba(255,255,255,0.03) !important; border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 16px !important; padding: 18px !important; }
[data-testid="stMetricLabel"] { color: #3d3d5c !important; font-size: 12px !important; font-weight: 700 !important; letter-spacing: 1px !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { color: #e2e2ef !important; font-family: 'Space Grotesk', sans-serif !important; font-size: 2rem !important; }

/* FILE UPLOADER */
.stFileUploader > div { background: rgba(255,255,255,0.02) !important; border: 2px dashed rgba(99,102,241,0.28) !important; border-radius: 18px !important; padding: 2.5rem !important; transition: all 0.3s !important; }
.stFileUploader > div:hover { border-color: rgba(99,102,241,0.55) !important; background: rgba(99,102,241,0.05) !important; }

/* SELECTBOX */
.stSelectbox > div > div { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 12px !important; color: #e2e2ef !important; }

/* DOWNLOAD */
.stDownloadButton > button { background: rgba(16,185,129,0.1) !important; border: 1px solid rgba(16,185,129,0.3) !important; color: #34d399 !important; border-radius: 12px !important; font-weight: 700 !important; padding: 10px 22px !important; }
.stDownloadButton > button:hover { background: rgba(16,185,129,0.2) !important; transform: translateY(-1px) !important; }

/* FOOTER */
.footer { text-align: center; padding: 2.5rem 0 1rem; border-top: 1px solid rgba(255,255,255,0.05); margin-top: 3rem; color: #2a2a40; font-size: 13px; }
.footer a { color: #6366f1; text-decoration: none; font-weight: 700; }
.footer a:hover { color: #818cf8; }
hr { border-color: rgba(255,255,255,0.05) !important; }
</style>

<div class="orb orb1"></div>
<div class="orb orb2"></div>
<div class="orb orb3"></div>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────
def clean_text(text):
    t = re.sub(r'http\S+|@\w+|#\w+', '', str(text))
    return t.strip()

def analyze_sentiment(text):
    blob = TextBlob(clean_text(text))
    pol = blob.sentiment.polarity
    sub = blob.sentiment.subjectivity
    if pol > 0.1:
        label, emoji, css = "Positive", "😊", "pos"
        conf = min(100, int(50 + pol * 50))
    elif pol < -0.1:
        label, emoji, css = "Negative", "😔", "neg"
        conf = min(100, int(50 + abs(pol) * 50))
    else:
        label, emoji, css = "Neutral", "😐", "neu"
        conf = max(50, int(72 - abs(pol) * 80))
    return {"label": label, "emoji": emoji, "css": css,
            "polarity": round(pol, 3), "subjectivity": round(sub, 3), "confidence": conf}

POS_WORDS = {'great','good','love','excellent','amazing','fantastic','wonderful','best',
             'awesome','perfect','happy','brilliant','superb','outstanding','recommend'}
NEG_WORDS = {'bad','terrible','awful','horrible','worst','poor','disappointing','hate',
             'useless','broken','wrong','waste','failed','annoying','slow','damaged'}

def get_keywords(text):
    words = set(re.findall(r'\b\w+\b', text.lower()))
    return sorted(words & POS_WORDS), sorted(words & NEG_WORDS)


# ════════════════════════════════════════
# HERO
# ════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-badge"><span class="dot"></span>AI-Powered NLP Analysis</div>
    <h1 class="hero-title">Understand What Your<br><span class="grad">Customers Really Feel</span></h1>
    <p class="hero-sub">Instant sentiment intelligence for any text — reviews, tweets, feedback, and more.</p>
    <div class="pills">
        <div class="pill"><span class="pill-icon">⚡</span>Real-time Results</div>
        <div class="pill"><span class="pill-icon">📊</span>Bulk CSV Support</div>
        <div class="pill"><span class="pill-icon">🎯</span>Confidence Scoring</div>
        <div class="pill"><span class="pill-icon">🔑</span>Keyword Detection</div>
        <div class="pill"><span class="pill-icon">📈</span>Visual Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["  🔍  Analyze Single Review  ", "  📂  Bulk CSV Analysis  "])

# ════════════════════════════════════════
# TAB 1
# ════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    L, R = st.columns([1.1, 0.9], gap="large")

    with L:
        st.markdown('<div class="sec-title">Paste Your Text</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-sub">Enter any review, tweet, or customer feedback</div>', unsafe_allow_html=True)

        user_input = st.text_area("", height=210,
            placeholder="Type or paste text here...\n\nTry: 'Absolutely loved this product! Fast shipping and great packaging.'",
            key="txt", label_visibility="collapsed")

        if "inject" in st.session_state:
            user_input = st.session_state.pop("inject")

        st.markdown('<div class="ex-label">⚡ Try Quick Examples</div>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        if b1.button("😊 Positive", use_container_width=True):
            st.session_state["inject"] = "Absolutely love this product! Best purchase I've ever made. Super fast delivery and quality is outstanding!"
            st.rerun()
        if b2.button("😔 Negative", use_container_width=True):
            st.session_state["inject"] = "Terrible quality. The item broke after one day. Customer service was horrible. Waste of money."
            st.rerun()
        if b3.button("😐 Neutral", use_container_width=True):
            st.session_state["inject"] = "The package arrived on Wednesday. It contained the product I ordered. It works as described."
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        go = st.button("🧠  Analyze Sentiment", use_container_width=True, type="primary")

    with R:
        if go and user_input.strip():
            with st.spinner("Analyzing..."):
                time.sleep(0.5)
            r = analyze_sentiment(user_input)
            css_map = {"pos": "rc-pos", "neg": "rc-neg", "neu": "rc-neu"}
            cf_map = {"pos": "cf-p", "neg": "cf-n", "neu": "cf-u"}

            st.markdown(f"""
            <div class="result-card {css_map[r['css']]}">
                <span class="rc-emoji">{r['emoji']}</span>
                <div class="rc-label">{r['label']}</div>
                <div class="rc-sub">Sentiment detected via NLP analysis</div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="conf-wrap">
                <div class="conf-head">
                    <span class="conf-title">Confidence Score</span>
                    <span class="conf-num">{r['confidence']}%</span>
                </div>
                <div class="conf-track">
                    <div class="{cf_map[r['css']]}" style="width:{r['confidence']}%"></div>
                </div>
            </div>""", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            pol_color = "#10b981" if r['polarity'] > 0 else "#ef4444" if r['polarity'] < 0 else "#94a3b8"
            c1.markdown(f"""<div class="mg">
                <div class="mg-label">Polarity</div>
                <div class="mg-val" style="color:{pol_color}">{r['polarity']}</div>
                <div class="mg-desc">-1.0 negative → +1.0 positive</div>
            </div>""", unsafe_allow_html=True)
            c2.markdown(f"""<div class="mg">
                <div class="mg-label">Subjectivity</div>
                <div class="mg-val">{r['subjectivity']}</div>
                <div class="mg-desc">0 factual → 1.0 opinion</div>
            </div>""", unsafe_allow_html=True)

            pos_kw, neg_kw = get_keywords(user_input)
            if pos_kw or neg_kw:
                tags = "".join([f'<span class="kw-p">✓ {w}</span>' for w in pos_kw])
                tags += "".join([f'<span class="kw-n">✗ {w}</span>' for w in neg_kw])
                st.markdown(f'<div class="kw-wrap"><div class="kw-title">Key Words Detected</div><div class="kw-tags">{tags}</div></div>', unsafe_allow_html=True)

        elif go:
            st.warning("⚠️  Please enter some text to analyze.")
        else:
            st.markdown("""<div class="empty-state">
                <span class="empty-icon">🧠</span>
                <p class="empty-text">Your analysis will appear here</p>
                <p style="font-size:13px;color:#5a5a7a;margin-top:6px">Enter text and click Analyze</p>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════
# TAB 2
# ════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Bulk Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Upload a CSV and analyze hundreds of reviews at once — with full visual reports.</div>', unsafe_allow_html=True)

    sample = pd.DataFrame({"review": [
        "Absolutely love this! Best purchase ever.",
        "Terrible quality. Broke after one day.",
        "The item arrived. Works as expected.",
        "Amazing service! Problem resolved instantly.",
        "Not worth the price at all. Very disappointed.",
        "Decent product. Does the job.",
        "Blown away by quality! Will buy again.",
        "Poor packaging. Item arrived damaged.",
    ]})
    st.download_button("⬇️ Download Sample CSV", sample.to_csv(index=False), "sample_reviews.csv", "text/csv")
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop your CSV here", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"✅  {len(df)} rows loaded")
        col = st.selectbox("Select text column:", df.columns.tolist())

        if st.button("⚡  Analyze All Reviews", type="primary"):
            prog = st.progress(0, text="Starting analysis...")
            results = []
            for i, txt in enumerate(df[col].astype(str)):
                results.append(analyze_sentiment(txt))
                prog.progress((i+1)/len(df), text=f"Analyzing {i+1} / {len(df)}...")
                time.sleep(0.008)
            prog.empty()

            df["Sentiment"]    = [r["label"]       for r in results]
            df["Polarity"]     = [r["polarity"]     for r in results]
            df["Subjectivity"] = [r["subjectivity"] for r in results]
            df["Confidence"]   = [r["confidence"]   for r in results]
            counts = df["Sentiment"].value_counts()

            st.markdown("<br>", unsafe_allow_html=True)
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("📋 Total", len(df))
            m2.metric("😊 Positive", counts.get("Positive",0))
            m3.metric("😔 Negative", counts.get("Negative",0))
            m4.metric("😐 Neutral",  counts.get("Neutral",0))

            st.markdown("<br>", unsafe_allow_html=True)
            COLORS = {"Positive":"#10b981","Negative":"#ef4444","Neutral":"#6b7280"}
            ch1, ch2 = st.columns(2)

            with ch1:
                fig, ax = plt.subplots(figsize=(5,4.5))
                fig.patch.set_facecolor('#0d0d14'); ax.set_facecolor('#0d0d14')
                cols = [COLORS.get(l,"#6b7280") for l in counts.index]
                wedges, texts, autos = ax.pie(counts.values, labels=counts.index, colors=cols,
                    autopct='%1.1f%%', startangle=140, pctdistance=0.72,
                    wedgeprops=dict(width=0.55, edgecolor='#0d0d14', linewidth=3))
                for t in texts: t.set_color('#9ca3af'); t.set_fontsize(13); t.set_fontweight('700')
                for a in autos: a.set_color('white'); a.set_fontsize(12); a.set_fontweight('700')
                ax.set_title("Sentiment Breakdown", color='#e2e2ef', fontsize=14, fontweight='800', pad=20)
                st.pyplot(fig); plt.close()

            with ch2:
                fig, ax = plt.subplots(figsize=(5,4.5))
                fig.patch.set_facecolor('#0d0d14'); ax.set_facecolor('#0d0d14')
                cols = [COLORS.get(l,"#6b7280") for l in counts.index]
                bars = ax.bar(counts.index, counts.values, color=cols, width=0.45, edgecolor='none', zorder=3)
                for b, v in zip(bars, counts.values):
                    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.1, str(v),
                            ha='center', color='white', fontweight='800', fontsize=14)
                ax.tick_params(colors='#6b7280', labelsize=13)
                ax.set_title("Review Counts", color='#e2e2ef', fontsize=14, fontweight='800', pad=20)
                ax.yaxis.grid(True, color='#1a1a2a', linewidth=0.8, zorder=0)
                for sp in ax.spines.values(): sp.set_visible(False)
                ax.set_ylabel("Count", color='#3d3d5c', fontsize=11)
                st.pyplot(fig); plt.close()

            # Polarity histogram
            fig, ax = plt.subplots(figsize=(10, 3.5))
            fig.patch.set_facecolor('#0d0d14'); ax.set_facecolor('#0d0d14')
            pm = df["Polarity"]>0.1; nm = df["Polarity"]<-0.1; um = ~(pm|nm)
            if pm.any(): ax.hist(df.loc[pm,"Polarity"], bins=15, color='#10b981', alpha=0.85, label='Positive', edgecolor='none')
            if nm.any(): ax.hist(df.loc[nm,"Polarity"], bins=15, color='#ef4444', alpha=0.85, label='Negative', edgecolor='none')
            if um.any(): ax.hist(df.loc[um,"Polarity"], bins=5,  color='#6b7280', alpha=0.85, label='Neutral',  edgecolor='none')
            ax.axvline(0, color='#e2e2ef', linewidth=1.5, linestyle='--', alpha=0.4)
            ax.set_title("Polarity Score Distribution", color='#e2e2ef', fontsize=14, fontweight='800', pad=14)
            ax.tick_params(colors='#6b7280', labelsize=11)
            ax.legend(facecolor='#1a1a2a', edgecolor='none', labelcolor='white', fontsize=12)
            ax.yaxis.grid(True, color='#1a1a2a', linewidth=0.8)
            for sp in ax.spines.values(): sp.set_visible(False)
            st.pyplot(fig); plt.close()

            st.markdown("<br>**📋 Full Results Table**")
            st.dataframe(df, use_container_width=True, height=320)
            st.download_button("⬇️ Export Results CSV", df.to_csv(index=False), "sentiment_results.csv", "text/csv")

# ── Footer ─────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <strong>SentimentAI</strong> — Built by
    <a href="https://github.com/MuhammadZafran33" target="_blank">Muhammad Zafran</a> ·
    Powered by TextBlob & Streamlit ·
    <a href="https://fiverr.com/muh_zafran" target="_blank">Hire me on Fiverr</a>
</div>
""", unsafe_allow_html=True)
