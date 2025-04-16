# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from pathlib import Path
import streamlit as st

# Load IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load model
path = Path('Model/Model.h5')
if not path.exists():
    raise FileNotFoundError(f"Model file not found at {path}")
model = load_model(path)

# Step 2: Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [min(word_index.get(word, 2) + 3, 9999) for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Step 3: Streamlit UI
st.set_page_config(page_title="IMDB Sentiment Classifier", page_icon="🎬", layout="centered")
st.markdown("## 🎬 IMDB Movie Review Sentiment Analysis")
st.markdown("Enter a movie review below, and the model will predict if it's **Positive** or **Negative**.")

# Input area
user_input = st.text_area("✍️ Movie Review", height=150, placeholder="Type or paste your review here...")

# Classification
if st.button("🔍 Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a movie review to classify.")
    else:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)[0][0]
        sentiment = "😊 Positive" if prediction > 0.5 else "😞 Negative"
        
        # Display result
        st.markdown("---")
        st.subheader("🔎 Prediction Result")
        st.markdown(f"**Sentiment:** {sentiment}")
        st.markdown(f"**Confidence Score:** `{prediction:.4f}`")
        st.markdown("---")
else:
    st.info("Awaiting input...")

