import streamlit as st
import prediction.prediction_scratch as pred


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
        # Call the prediction function
        sentiment, prediction = pred.predict_sentiment(user_input)
        
        
        
        # Display result
        st.markdown("---")
        st.subheader("🔎 Prediction Result")
        st.markdown(f"**Sentiment:** {sentiment}")
        st.markdown(f"**Confidence Score:** `{prediction:.4f}`")
        st.markdown("---")
else:
    st.info("Awaiting input...")