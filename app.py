import streamlit as st
import joblib
from backend import clean_tweet, nltk_preprocess
from scipy.sparse import hstack
import joblib
import time

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
    model = joblib.load("best_emotion_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

# Function to predict emotion from user input
def predict_emotion(input_text, model, vectorizer):
    # Clean and preprocess input text
    cleaned_text = clean_tweet(input_text)
    processed_text = nltk_preprocess(cleaned_text)
    
    # Vectorize the input text
    vectorized_text = vectorizer.transform([processed_text])
    
    # Add a placeholder for the intensity feature
    intensity_placeholder = np.array([[0]])  # Replace '0' with actual intensity if available
    combined_features = hstack((vectorized_text, intensity_placeholder))  # Combine features
    
    # Predict using the model
    prediction = model.predict(combined_features)
    return prediction[0]



# Streamlit app interface
st.title("Emotion Detection from Text")

st.subheader("Enter a text to analyze its emotion")
input_text = st.text_area("Your input text:", "")

if input_text:
    with st.spinner("Predicting emotion..."):

        model, vectorizer = load_model_and_vectorizer()
        predicted_emotion = predict_emotion(input_text, model, vectorizer)

    # Map numeric prediction back to emotion
    emotion_mapping = {0: "anger", 1: "fear", 2: "joy", 3: "sadness"}
    st.write(f"The predicted emotion is: **{emotion_mapping[predicted_emotion]}**")
