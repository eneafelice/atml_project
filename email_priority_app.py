import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="📬 Email Priority Analyzer", layout="centered")

# Load NLP models (cached for performance)
@st.cache_resource
def load_models():
    sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    emotion = pipeline("text-classification", model="nateraw/bert-base-uncased-emotion", top_k=None)
    return sentiment, emotion

sentiment_model, emotion_model = load_models()

# Define urgency-related keywords
urgency_words = [
    "urgent", "asap", "immediately", "right away",
    "as soon as possible", "can't access", "refund", "problem", "crash", "not working"
]

def get_sentiment_score(text):
    label = sentiment_model(text)[0]['label'].lower()
    return {'negative': 1.0, 'neutral': 0.5, 'positive': 0.0}.get(label, 0.5)

def get_emotion_score(text):
    emotions = emotion_model(text)[0]
    score = sum(e['score'] for e in emotions if e['label'].lower() in ['anger', 'fear', 'sadness'])
    return min(score, 1.0)

def get_urgency_score(text):
    return min(sum(word in text.lower() for word in urgency_words) / len(urgency_words), 1.0)

def compute_priority(sentiment, emotion, urgency, vip):
    vip_bonus = 0.2 if vip else 0
    score = (urgency * 0.4) + (emotion * 0.3) + (sentiment * 0.3) + vip_bonus
    return round(min(score, 1.0) * 100)

# ---- Streamlit UI ----
st.set_page_config(page_title="📬 Email Priority Analyzer", layout="centered")
st.title("📬 Smart Email Priority Analyzer")
st.markdown("Analyze urgency, emotion, and sentiment of customer emails to assign priority.")

email_text = st.text_area("✉️ Paste customer email here:", height=200)
vip_status = st.checkbox("💎 Mark as VIP customer", value=False)

if st.button("🚦 Analyze Priority") and email_text.strip():
    sentiment = get_sentiment_score(email_text)
    emotion = get_emotion_score(email_text)
    urgency = get_urgency_score(email_text)
    priority_score = compute_priority(sentiment, emotion, urgency, vip_status)

    st.subheader("📊 Priority Score")
    st.metric("🚨 Priority", f"{priority_score} / 100")

    with st.expander("🔍 Breakdown"):
        st.write(f"**Sentiment Score**: {sentiment:.2f} (1 = Negative, 0 = Positive)")
        st.write(f"**Negative Emotion Score**: {emotion:.2f}")
        st.write(f"**Urgency Score**: {urgency:.2f}")
        st.write(f"**VIP Bonus Applied**: {'✅ Yes' if vip_status else '❌ No'}")

    if priority_score > 75:
        st.success("🚨 High Priority — escalate immediately.")
    elif priority_score > 40:
        st.warning("⚠️ Medium Priority — respond soon.")
    else:
        st.info("✅ Low Priority — can wait.")
