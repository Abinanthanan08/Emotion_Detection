from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect
from deep_translator import GoogleTranslator
from langcodes import Language

# Load sentiment (RoBERTa)
sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)

# Load emotion detection (GoEmotions)
emotion_tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/bert-base-go-emotion")
emotion_model = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/bert-base-go-emotion")
emotion_pipeline = pipeline("text-classification", model=emotion_model, tokenizer=emotion_tokenizer, return_all_scores=True)

sentiment_labels = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

emotion_emojis = {
    "admiration": "👏", "joy": "😊", "sadness": "😢", "anger": "😠", "fear": "😨", "surprise": "😲",
    "disapproval": "👎", "love": "❤️", "neutral": "😐", "gratitude": "🙏", "realization": "💡"
}

def detect_and_translate(text):
    lang = detect(text)
    if lang != "en":
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated, lang
    return text, lang

def predict_emotion(text):
    result = emotion_pipeline(text)[0]
    result.sort(key=lambda x: x['score'], reverse=True)
    labels = [r["label"] for r in result]
    scores = [r["score"] for r in result]
    return labels[0], scores[0], labels, scores

def get_sentiment(text):
    output = sentiment_pipeline(text)[0]
    sentiment = sentiment_labels.get(output["label"], "Unknown")
    polarity = {
        "Negative": -1.0,
        "Neutral": 0.0,
        "Positive": 1.0
    }.get(sentiment, 0.0)
    return sentiment, polarity

def get_language_name(code):
    try:
        return Language.get(code).display_name().capitalize()
    except:
        return code  

