# 💬 SentimentAI — Customer Sentiment Analysis App

<div align="center">

![SentimentAI Banner](https://img.shields.io/badge/SentimentAI-Text%20Intelligence-6366f1?style=for-the-badge&logo=python&logoColor=white)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://zafran-sentiment-analysis-app.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![TextBlob](https://img.shields.io/badge/TextBlob-NLP-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

**Understand what your customers really feel — instantly.**

[🚀 Live Demo](https://zafran-sentiment-analysis-app.streamlit.app) · [👤 Hire Me on Fiverr](https://fiverr.com/muh_zafran) · [📁 Portfolio](https://github.com/MuhammadZafran33)

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Single Review Analysis** | Paste any text and get instant Positive / Negative / Neutral result |
| 📊 **Confidence Score** | Shows how confident the model is in its prediction (0–100%) |
| 🎯 **Polarity & Subjectivity** | Detailed NLP scores for deeper text understanding |
| 🔑 **Keyword Detection** | Highlights positive and negative words in your text |
| 📂 **Bulk CSV Analysis** | Upload hundreds of reviews and analyze all at once |
| 📈 **Visual Analytics** | Pie chart, bar chart, and polarity distribution histogram |
| ⬇️ **Export Results** | Download analyzed results as CSV |
| 🌙 **Premium Dark UI** | Professional glassmorphism design with animated background |

---

## 🖥️ Live Demo

🔗 **[https://zafran-sentiment-analysis-app.streamlit.app](https://zafran-sentiment-analysis-app.streamlit.app)**

### Single Review Mode
> Paste any customer review → click Analyze → get instant sentiment with confidence score and keyword highlights

### Bulk CSV Mode
> Upload a CSV with a review column → analyze all rows → view charts → download results

---

## 🛠️ Tech Stack

```
Python 3.9+
├── Streamlit        → Web app framework
├── TextBlob         → NLP sentiment analysis
├── Pandas           → Data manipulation
├── Matplotlib       → Data visualization
└── Regex            → Text cleaning
```

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/MuhammadZafran33/sentiment-analysis-app.git
cd sentiment-analysis-app
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## 📦 Requirements

```txt
streamlit
textblob
pandas
matplotlib
nltk
```

---

## 📊 How It Works

```
User Input Text
      │
      ▼
Text Cleaning (remove URLs, mentions, hashtags)
      │
      ▼
TextBlob NLP Analysis
      │
      ├── Polarity Score  (-1.0 to +1.0)
      └── Subjectivity    (0.0 to 1.0)
            │
            ▼
      Classification
      ├── > 0.1  → 😊 Positive
      ├── < -0.1 → 😔 Negative
      └── else   → 😐 Neutral
            │
            ▼
      Confidence Score + Keyword Detection + Visualization
```

---

## 📁 Project Structure

```
sentiment-analysis-app/
│
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 🎯 Use Cases

- 🛍️ **E-commerce** — Analyze product reviews at scale
- 🍕 **Restaurants** — Understand customer feedback
- 📱 **Mobile Apps** — Monitor user ratings and comments
- 🐦 **Social Media** — Track brand sentiment on Twitter
- 📧 **Customer Support** — Prioritize negative tickets automatically
- 🏥 **Healthcare** — Analyze patient feedback

---

## 📸 Screenshots

### Hero Section
> Premium dark UI with gradient title and feature pills

### Single Review Analysis
> Real-time prediction with confidence bar, polarity meter, and keyword badges

### Bulk CSV Analysis
> Upload CSV → progress bar → metrics → pie chart + bar chart + histogram → export

---

## 🤝 Hire Me

Need a custom ML web app for your business?

[![Fiverr](https://img.shields.io/badge/Hire%20Me-Fiverr-1DBF73?style=for-the-badge&logo=fiverr&logoColor=white)](https://fiverr.com/muh_zafran)

**Services I offer:**
- ✅ Custom ML Streamlit apps
- ✅ Sentiment analysis for your dataset
- ✅ Classification & regression models
- ✅ Data visualization dashboards
- ✅ Model deployment on Streamlit Cloud

---

## 👨‍💻 Author

**Muhammad Zafran**
- GitHub: [@MuhammadZafran33](https://github.com/MuhammadZafran33)
- Fiverr: [muh_zafran](https://fiverr.com/muh_zafran)

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">
  <strong>⭐ If you found this useful, please star the repo!</strong><br><br>
  Built with ❤️ using Python & Streamlit
</div>
