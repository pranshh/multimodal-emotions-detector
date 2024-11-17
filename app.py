import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import joblib
import numpy as np

# Download necessary NLTK packages
nltk.download('stopwords')
nltk.download('punkt')

# Function to clean the tweet text
def clean_tweet(tweet):
    tweet = re.sub(r"(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", str(tweet))
    tweet = re.sub(r"\s+", " ", tweet).strip()
    return tweet

# Function to preprocess the text using NLTK
def nltk_preprocess(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(filtered_words)

# Function to load the trained model and vectorizer
def load_model_and_vectorizer():
    model = joblib.load("best_emotion_model.pkl")  # Load the best trained model
    vectorizer = joblib.load("vectorizer.pkl")  # Load the vectorizer
    return model, vectorizer

# Function to preprocess and predict the emotion from the input text
def predict_emotion(input_text, model, vectorizer):
    # Clean and preprocess input text
    cleaned_text = clean_tweet(input_text)
    processed_text = nltk_preprocess(cleaned_text)

    # Vectorize the input text
    vectorized_text = vectorizer.transform([processed_text])

    # Add a placeholder for the intensity feature (set to 0 as a default)
    intensity_placeholder = np.array([[0]])  # Replace '0' with actual intensity if available
    combined_features = hstack((vectorized_text, intensity_placeholder))  # Combine features

    # Predict using the model
    prediction = model.predict(combined_features)
    return prediction[0]

# Streamlit User Interface
st.title("Emotion Detection from Text")

# Load the trained model and vectorizer
model, vectorizer = load_model_and_vectorizer()

# Input for text
st.subheader("Enter a sentence to predict its emotion:")
input_text = st.text_area("Text Input", "")

# Predict button
if st.button("Predict Emotion"):
    if input_text.strip():
        with st.spinner('Predicting emotion...'):
            try:
                # Predict the emotion
                predicted_emotion = predict_emotion(input_text, model, vectorizer)
                st.write(f"The predicted emotion is: **{predicted_emotion}**")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter some text for prediction.")
