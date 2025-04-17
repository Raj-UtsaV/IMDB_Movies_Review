import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from pathlib import Path
import pickle
import os

def build_path(*path_parts):
    base_dir = Path.cwd().parent
    return os.path.join(base_dir, *path_parts)

# Load the pre-trained model with ReLU activation
path = Path(build_path('IMDB','Model','Model_scratch.h5'))
if not path.exists():
    raise FileNotFoundError(f"Model file not found at {path}")
model = load_model(path)

# Load the tokenizer
with open(build_path('IMDB', 'Encoder', 'tokenizer.pickle'), 'rb') as handle:
    tokenizer = pickle.load(handle)
    
# Function to preprocess user input
def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    
    # Pad sequence to fixed length of 400
    padded_review = sequence.pad_sequences(sequences, maxlen=400)
    
    return padded_review

### Prediction  function

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction=model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    return sentiment, prediction[0][0]

### Prediction  function
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction=model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    return sentiment, prediction[0][0]

# example_review = "This is a terrible movie. The acting was bad and the plot made no sense at all."
# sentiment, score = predict_sentiment(example_review)

# print(f'Review: {example_review}')
# print(f'Sentiment: {sentiment}')
# print(f'Prediction Score: {score}')