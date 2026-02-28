import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from textblob import TextBlob
import re

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stTextArea textarea {
        background-color: #1e2130;
        color: white;
        border: 1px solid #4a90d9;
        border-radius: 8px;
        font-size: 15px;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin: 10px 0;
    }
    .positive { background: linear-gradient(135deg, #1a4a2e, #2d7a4f); border: 1px solid #2d7a4f; }
    .negative { background: linear-gradient(135deg, #4a1a1a, #7a2d2d); border: 1px solid #7a2d2d; }
    .neutral  { background: linear-gradient(135deg, #2a2a1a, #5a5a2d); border: 1px solid #5a5a2d; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #2d3250;
    }
</style>
""", unsafe_allow_html=True)

# ── Helper functions ─────────────────────────────────────────
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    return text.strip()

def analyze_sentiment(text):
    cleaned = clean_text(text)
    blob = TextBlob(cleaned)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.1:
        label = "Positive"
        emoji = "😊"
        confidence = min(100, int(50 + polarity * 50))
    elif polarity < -0.1:
        label = "Negative"
        emoji = "😞"
        confidence = min(100, int(50 + abs(polarity) * 50))
    else:
        label = "Neutral"
        emoji = "😐"
        confidence = int(70 - abs(polarity) * 100)

    return {
        "label": label,
        "emoji": emoji,
        "polarity": round(polarity, 3),
        "subjectivity": round(subjectivity, 3),
        "confidence": confidence
    }

def get_css_class(label):
    return label.lower()

# ── Header ───────────────────────────────────────────────────
st.markdown("## 💬 Customer Sentiment Analysis App")
st.markdown("Analyze the emotion behind any text — single review or bulk CSV upload.")
st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📝 Single Review", "📂 Bulk CSV Analysis"])

# ════════════════════════════════════════════
# TAB 1 — Single Review
# ════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("### Enter your text below")
        user_input = st.text_area(
            label="",
            placeholder="e.g. The product quality was amazing! I loved every bit of it.",
            height=180,
            key="single_input"
        )

        examples = {
            "😊 Positive example": "The delivery was super fast and the product exceeded my expectations!",
            "😞 Negative example": "Terrible experience. The item broke after one day. Very disappointed.",
            "😐 Neutral example": "The package arrived on Tuesday. It contains the items I ordered."
        }
        st.markdown("**Try an example:**")
        ecol1, ecol2, ecol3 = st.columns(3)
        for col, (label, text) in zip([ecol1, ecol2, ecol3], examples.items()):
            if col.button(label, use_container_width=True):
                user_input = text

        analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary")

    with col2:
        if analyze_btn and user_input.strip():
            result = analyze_sentiment(user_input)
            css = get_css_class(result["label"])

            st.markdown(f"""
            <div class="result-box {css}">
                <h1 style="font-size:60px; margin:0">{result['emoji']}</h1>
                <h2 style="color:white; margin:8px 0">{result['label']}</h2>
                <p style="color:#ccc; font-size:14px">Confidence: {result['confidence']}%</p>
            </div>
            """, unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <p style="color:#aaa; font-size:12px; margin:0">POLARITY</p>
                    <h3 style="color:#4a90d9; margin:4px 0">{result['polarity']}</h3>
                    <p style="color:#666; font-size:11px">-1 (neg) to +1 (pos)</p>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <p style="color:#aaa; font-size:12px; margin:0">SUBJECTIVITY</p>
                    <h3 style="color:#4a90d9; margin:4px 0">{result['subjectivity']}</h3>
                    <p style="color:#666; font-size:11px">0 (objective) to 1 (subjective)</p>
                </div>""", unsafe_allow_html=True)

            # Polarity bar
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Sentiment Scale**")
            fig, ax = plt.subplots(figsize=(5, 0.6))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#1e2130')
            ax.barh(0, 2, left=-1, color='#2d3250', height=0.5)
            ax.barh(0, result['polarity'], color='#4a90d9' if result['polarity'] >= 0 else '#d94a4a', height=0.5)
            ax.axvline(0, color='white', linewidth=1)
            ax.set_xlim(-1, 1)
            ax.set_yticks([])
            ax.set_xticks([-1, -0.5, 0, 0.5, 1])
            ax.tick_params(colors='white', labelsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
            st.pyplot(fig)
            plt.close()

        elif analyze_btn:
            st.warning("Please enter some text to analyze.")
        else:
            st.markdown("""
            <div style="text-align:center; color:#555; padding:60px 20px;">
                <h1>💬</h1>
                <p>Enter text on the left and click Analyze</p>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════
# TAB 2 — Bulk CSV Analysis
# ════════════════════════════════════════════
with tab2:
    st.markdown("### Upload a CSV file with a text/review column")

    sample_df = pd.DataFrame({
        "review": [
            "Great product, highly recommend!",
            "Worst purchase ever. Total waste of money.",
            "It was okay. Nothing special.",
            "Absolutely love it! Will buy again.",
            "Delivery was late and packaging was damaged."
        ]
    })

    csv_data = sample_df.to_csv(index=False)
    st.download_button("⬇️ Download Sample CSV", csv_data, "sample_reviews.csv", "text/csv")

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown(f"**{len(df)} rows loaded.** Select the text column:")

        text_col = st.selectbox("Text column", df.columns.tolist())

        if st.button("🔍 Analyze All Reviews", type="primary"):
            with st.spinner("Analyzing..."):
                results = df[text_col].astype(str).apply(analyze_sentiment)
                df["Sentiment"]    = results.apply(lambda x: x["label"])
                df["Polarity"]     = results.apply(lambda x: x["polarity"])
                df["Subjectivity"] = results.apply(lambda x: x["subjectivity"])
                df["Confidence"]   = results.apply(lambda x: x["confidence"])

            counts = df["Sentiment"].value_counts()

            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Reviews", len(df))
            c2.metric("😊 Positive", counts.get("Positive", 0))
            c3.metric("😞 Negative", counts.get("Negative", 0))
            c4.metric("😐 Neutral",  counts.get("Neutral", 0))

            # Charts
            ch1, ch2 = st.columns(2)

            with ch1:
                fig, ax = plt.subplots(figsize=(4, 4))
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#0e1117')
                colors = ['#2d7a4f', '#7a2d2d', '#7a7a2d']
                labels = counts.index.tolist()
                ax.pie(counts.values, labels=labels, colors=colors[:len(labels)],
                       autopct='%1.1f%%', textprops={'color': 'white'})
                ax.set_title("Sentiment Distribution", color='white')
                st.pyplot(fig)
                plt.close()

            with ch2:
                fig, ax = plt.subplots(figsize=(4, 4))
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#1e2130')
                bar_colors = ['#2d7a4f' if l == 'Positive' else '#7a2d2d' if l == 'Negative' else '#7a7a2d'
                              for l in counts.index]
                ax.bar(counts.index, counts.values, color=bar_colors)
                ax.set_facecolor('#1e2130')
                ax.tick_params(colors='white')
                ax.set_title("Sentiment Count", color='white')
                for spine in ax.spines.values():
                    spine.set_color('#2d3250')
                fig.patch.set_facecolor('#0e1117')
                st.pyplot(fig)
                plt.close()

            # Results table
            st.markdown("### 📋 Detailed Results")
            st.dataframe(df, use_container_width=True)

            # Download results
            st.download_button(
                "⬇️ Download Results CSV",
                df.to_csv(index=False),
                "sentiment_results.csv",
                "text/csv"
            )

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555; font-size:12px;'>"
    "Built by Muhammad Zafran · "
    "<a href='https://github.com/MuhammadZafran33' style='color:#4a90d9;'>GitHub</a> · "
    "Powered by TextBlob & Streamlit</p>",
    unsafe_allow_html=True
)
