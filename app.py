import streamlit as st
import joblib 
import re 
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words=set(stopwords.words("english"))

#load the model and vectorizer
model=joblib.load("sentimental_model.pkl")
vectorizer=joblib.load("vectorizer.pkl")

#clean text
def clean_text(text):
  text=re.sub(r"[^a-zA-Z]"," ",text).lower()
  tokens=text.split()
  tokens=[word for word in tokens if word not in stop_words]
  return " ".join(tokens)

#create UI
st.set_page_config(page_title="Sentimental Analysis for Movies",layout="centered")
st.title("Sentiment Analysis app for movies")
st.markdown("enter the review of a movie")

user_input=st.text_area("eneter the review ")

if st.button('Predict Sentiment'):
  cleaned=clean_text(user_input)
  vectorized=vectorizer.transform([cleaned])
  prediction=model.predict(vectorized)[0]
  sentiment="Positive" if prediction==1 else "Negative"
  st.success("Prediction is:"+ sentiment)
