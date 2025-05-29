import streamlit as st
import plotly.express as px
import pandas as pd
import time
import requests
from model import detect_and_translate, predict_emotion, get_sentiment, get_language_name

# Emojis for emotion display
emotion_emojis = {
    "joy": "😊", "sadness": "😢", "anger": "😠", "fear": "😨", "surprise": "😲",
    "disgust": "🤢", "love": "❤️", "neutral": "😐", "admiration": "👏",
    "gratitude": "🙏", "realization": "💡", "approval": "👍", "disapproval": "👎",
    "curiosity": "❓", "others": "🤔"
}

# ---------------- CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #e0f7fa, #fce4ec);
    font-family: 'Poppins', sans-serif;
    color: #333;
}
.title {
    font-size: 3.2rem;
    font-weight: 700;
    color: #008080;
    text-align: center;
    margin-bottom: 0.2rem;
    animation: fadeIn 2s ease-in-out;
}
.subtitle {
    font-size: 1.3rem;
    text-align: center;
    color: #444;
    margin-bottom: 2rem;
}
textarea {
    border: 2px solid #008080 !important;
    border-radius: 12px !important;
    font-size: 1.1rem !important;
    padding: 10px !important;
}
button[kind="primary"], .stButton>button {
    background-color: #008080 !important;
    color: white !important;
    border-radius: 10px !important;
    font-size: 1.1rem !important;
    font-weight: bold !important;
}
.loading-dots {
    font-weight: bold;
    font-size: 1.4rem;
    color: #008080;
    animation: blink 1.2s infinite;
}
.result-card {
    background: #fff;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    animation: fadeIn 1s ease-out;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #e0f7fa, #fff3e0);
    padding: 1.5rem;
    border-top-right-radius: 15px;
    border-bottom-right-radius: 15px;
    box-shadow: 2px 0 8px rgba(0,0,0,0.1);
}
@keyframes blink {
  0%, 20% {opacity: 0;}
  50% {opacity: 1;}
  100% {opacity: 0;}
}
@keyframes fadeIn {
  from {opacity: 0; transform: translateY(20px);}
  to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ---------------- UI ----------------
st.markdown('<div class="title">🤖 Multilingual Emotion & Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">💬 Analyze how people feel — in any language!</div>', unsafe_allow_html=True)

user_input = st.text_area("✍️ Enter your text:", height=150, placeholder="Type something in English, Tamil, Hindi, etc...")

if user_input.strip():
    loading_placeholder = st.empty()
    progress_bar = st.progress(0)

    for i in range(4):
        loading_placeholder.markdown(f"<div class='loading-dots'>Analyzing{'.' * i}</div>", unsafe_allow_html=True)
        time.sleep(0.5)
        progress_bar.progress((i + 1) * 25)

    loading_placeholder.empty()
    progress_bar.empty()

    # Language detection and processing
    translated_text, detected_lang = detect_and_translate(user_input)
    lang_name = get_language_name(detected_lang)
    emotion, emotion_prob, labels, probs = predict_emotion(translated_text)
    sentiment, polarity = get_sentiment(translated_text)

    # Main results display
    st.success("✅ Analysis Complete!")
    st.markdown(f"<div class='result-card'><b>🌐 Detected Language:</b> {lang_name}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-card'><b>🔁 Translated Text:</b> {translated_text}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-card'><b>🎭 Detected Emotion:</b> {emotion.capitalize()} {emotion_emojis.get(emotion.lower(), '🤔')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-card'><b>🧠 Confidence Score:</b> {emotion_prob:.2f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-card'><b>💡 Sentiment:</b> {sentiment} {'👍' if sentiment=='POSITIVE' else '👎' if sentiment=='NEGATIVE' else '😐'}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-card'><b>📉 Polarity Score:</b> {polarity:.2f}</div>", unsafe_allow_html=True)

    # Emotion probability chart
    df = pd.DataFrame({"Emotions": labels, "Probability": probs})
    fig = px.bar(df, x='Emotions', y='Probability', color='Emotions',
                 text_auto=False, color_discrete_sequence=px.colors.qualitative.Pastel)

    fig.update_layout(
        title="🎨 Emotion Probability Distribution (Bar Chart)",
        font=dict(family="Poppins", size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("📤 Export & Feedback")

        # CSV Export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name='emotion_analysis.csv',
            mime='text/csv'
        )

        # Feedback
        GOOGLE_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSdhxeYH_fgDSjAi9nMbx5BJes23_-XJBg1mHviSpgXgKeBM_g/formResponse"
        ENTRY_ID = "entry.1302998468"

        st.subheader("📝 Feedback")
        feedback_text = st.text_area("Your feedback:", height=100)

        if st.button("Submit Feedback"):
            if feedback_text.strip():
                data = {ENTRY_ID: feedback_text}
                response = requests.post(GOOGLE_FORM_URL, data=data)
                if response.status_code in [200, 302]:
                    st.success("🙏 Thank you! Your feedback was submitted.")