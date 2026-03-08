import streamlit as st
from transformers import pipeline
from keybert import KeyBERT

st.title("AI Metin Analiz Aracı")
st.info("İlk çalıştırmada özetleme modeli indirileceği için lütfen birkaç saniye bekleyin.")

@st.cache_resource
def load_models():
    
    summarizer = pipeline("text-generation", model="google/flan-t5-small")
    sentiment = pipeline("sentiment-analysis")
    kw_model = KeyBERT()
    return summarizer, sentiment, kw_model

summarizer, sentiment, kw_model = load_models()

text = st.text_area("Metin giriniz")

if st.button("Analiz İçin Basın"):

    if len(text) < 20:
        st.warning("Analiz için daha uzun bir metin giriniz!")
    else:

        summary_result = summarizer(text, max_new_tokens=60)
        summary_text = summary_result[0]["generated_text"]

        sentiment_result = sentiment(text)

        keywords = kw_model.extract_keywords(text, top_n=5)

        st.subheader("Özet")
        st.write(summary_text)

        st.subheader("Duygu Analizi")
        st.write(sentiment_result[0]["label"])

        st.subheader("Anahtar Kelimeler")
        st.write("/ ".join([word for word, score in keywords]))