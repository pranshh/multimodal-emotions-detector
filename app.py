import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import joblib
import librosa
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import requests

def download_model_from_drive(drive_url, destination):
    # Function to download model from Google Drive
    if os.path.exists(destination):
        st.write(f"{destination} already exists.")
        return
    st.write(f"Downloading model from {drive_url}...")
    response = requests.get(drive_url, stream=True)
    with open(destination, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
    st.write("Download completed.")

# URLs to your models on Google Drive
image_model_paths = {
    "VGG19": "https://drive.google.com/file/d/1kHOQDJum7vYBF5hWoG00LStGqk8Dp9-V/view?usp=drive_link",
    "MobileNet": "https://drive.google.com/file/d/14WFyz9dAAkg9wkm2Ct1Xoo-Z6O7rFloq/view?usp=sharing",
    "EfficientNetB7": "https://drive.google.com/file/d/1j421sEEyp4p4GkhVS_JCJHkKmY8OWCkb/view?usp=sharing",
    "DenseNet121": "https://drive.google.com/file/d/1y1B8Blq62rd0tezR3R_UIOJwtrWzdyW8/view?usp=sharing",
    "ResNet50": "https://drive.google.com/file/d/1cGpFKLFD4JZszKX-NwuDVyWp6GLgA_fJ/view?usp=sharing"
}

# Download models if not already downloaded
for model_name, drive_url in image_model_paths.items():
    download_model_from_drive(drive_url, image_model_paths[model_name])

# Load your pre-trained models from the directory


image_models = {name: load_model(path) for name, path in image_model_paths.items()}

# Load LSTM models for speech emotion detection
speech_model_paths = {
    "Spectrogram LSTM": "spectogram_lstm_model.keras",
    "LSTM": "lstm_model.keras",
    "MFCC LSTM": "mfcc_lstm_model.keras",
}

speech_models = {name: load_model(path) for name, path in speech_model_paths.items()}

# Define class labels
class_labels = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
speech_labels = {0: 'neutral', 1: 'disgust', 2: 'sad', 3: 'pleasant_surprise', 4: 'angry', 5: 'fear', 6: 'happy'}

# Function to preprocess the image
def preprocess_image(image, target_size=(48, 48)):
    image = image.resize(target_size)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Function to predict the emotion from the image using each model
def predict_emotions(image, models):
    predictions = {}
    for model_name, model in models.items():
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_labels[predicted_class]
        predictions[model_name] = predicted_label
    return predictions

# Function to preprocess and predict the emotion from text using each model
def clean_tweet(tweet):
    tweet = re.sub(r"(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", str(tweet))
    tweet = re.sub(r"\s+", " ", tweet).strip()
    return tweet

def nltk_preprocess(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(filtered_words)

def predict_emotion(input_text, model, vectorizer):
    cleaned_text = clean_tweet(input_text)
    processed_text = nltk_preprocess(cleaned_text)
    vectorized_text = vectorizer.transform([processed_text])
    intensity_placeholder = np.array([[0]])
    combined_features = hstack((vectorized_text, intensity_placeholder))
    prediction = model.predict(combined_features)
    return prediction[0]

# Function to extract features from speech
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft, axis=1)

    fft = np.fft.fft(y)
    fft_mean = np.mean(np.abs(fft))

    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    mfccs_mean = np.mean(mfccs, axis=1)

    spectrogram = librosa.amplitude_to_db(librosa.stft(y))
    spectrogram_mean = np.mean(spectrogram, axis=1)

    features = np.concatenate((chroma_stft_mean, [fft_mean], mfccs_mean, spectrogram_mean))
    return features.reshape((1, 1, -1))

# Function to predict the emotion from speech using each model
def predict_speech_emotions(file_path, models):
    features = extract_features(file_path)
    predictions = {}
    for model_name, model in models.items():
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = speech_labels[predicted_class]
        predictions[model_name] = predicted_label
    return predictions

# Streamlit User Interface
st.title("Multimodal Emotion Detector")

# Sidebar for selecting the mode of emotion detection
st.sidebar.title("Choose Input Mode")
option = st.sidebar.selectbox("Select the type of input you want to analyze:", ("Text", "Image", "Speech"))

if option == "Text":
    st.subheader("Emotion Detection from Text")

    # Load the trained model and vectorizer
    def load_model_and_vectorizer():
        model = joblib.load("best_emotion_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer

    model, vectorizer = load_model_and_vectorizer()

    # Load the label mapping
    label_mapping = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'sadness'}

    # Input for text
    st.subheader("Enter a sentence to predict its emotion:")
    input_text = st.text_area("Text Input", "")

    # Predict button
    if st.button("Predict Emotion"):
        if input_text.strip():
            with st.spinner('Predicting emotion...'):
                try:
                    predicted_emotion_idx = predict_emotion(input_text, model, vectorizer)
                    predicted_emotion = label_mapping[predicted_emotion_idx]
                    st.write(f"The predicted emotion is: **{predicted_emotion}**")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter some text for prediction.")

elif option == "Image":
    st.subheader("Emotion Detection from Image")
    
    # Create an upload field for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and preprocess the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image for the models
        processed_image = preprocess_image(image)

        # Predict the emotion using each model
        predictions = predict_emotions(processed_image, image_models)
        
        # Display the predictions
        for model_name, predicted_label in predictions.items():
            st.write(f"{model_name} Prediction: **{predicted_label}**")

elif option == "Speech":
    st.subheader("Emotion Detection from Speech")

    # Create an upload field for audio file
    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_audio_file.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Predict the emotion using each model
        predictions = predict_speech_emotions("temp_audio_file.wav", speech_models)
        
        # Display the predictions
        for model_name, predicted_label in predictions.items():
            st.write(f"{model_name} Prediction: **{predicted_label}**")

