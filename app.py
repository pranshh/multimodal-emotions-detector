import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.sparse import hstack
import joblib
import time

# Download necessary NLTK packages
nltk.download('stopwords')
nltk.download('punkt_tab')

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
    cleaned_text = clean_tweet(input_text)  # Clean the input text
    processed_text = nltk_preprocess(cleaned_text)  # Preprocess using NLTK
    vectorized_text = vectorizer.transform([processed_text])  # Vectorize the text
    prediction = model.predict(vectorized_text)  # Get the predicted emotion
    return prediction[0]

# Streamlit User Interface
st.title("Emotion Detection from Text")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Preprocess the text data
    st.subheader("Preprocessing the text data")
    with st.spinner('Preprocessing your dataset...'):
        df['cleaned_tweet'] = df['text'].apply(clean_tweet)
        df['processed_text'] = df['cleaned_tweet'].apply(nltk_preprocess)
        st.write(df[['text', 'processed_text']].head())

    # Vectorize the text
    vectorizer = CountVectorizer()
    X_text = vectorizer.fit_transform(df['processed_text'])
    X_intensity = np.array(df['intensity']).reshape(-1, 1)
    X = hstack((X_text, X_intensity))

    # Mapping emotions to numeric values
    label_mapping = {'anger': 0, 'fear': 1, 'joy': 2, 'sadness': 3}
    df['numeric_emotion'] = df['emotion'].map(label_mapping)
    y = df['numeric_emotion']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }

    best_model = None
    best_accuracy = 0

    # Training and Evaluation
    for name, model in models.items():
        st.subheader(f"Training {name}")
        with st.spinner(f'Training {name}...'):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"{name} Accuracy: {accuracy * 100:.2f}%")

            # If this model has better accuracy, save it as the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model

            # Classification Report
            st.subheader(f"{name} Classification Report")
            st.text(classification_report(y_test, y_pred, target_names=list(label_mapping.keys())))

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))  # Create a new figure and axis for the confusion matrix plot
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_mapping.keys()))
            disp.plot(cmap=plt.cm.Blues, ax=ax)
            st.pyplot(fig)  # Pass the figure to st.pyplot to display it

            # Save the model and vectorizer
            joblib.dump(model, "emotion_model.pkl")
            joblib.dump(vectorizer, "vectorizer.pkl")

    # Save the best model
    if best_model is not None:
        st.write(f"Saving the best model with accuracy: {best_accuracy * 100:.2f}%")
        joblib.dump(best_model, "best_emotion_model.pkl")  # Save the best model

    # # Prediction on User Input
    # st.subheader("Emotion Prediction for Input Text")
    # input_text = st.text_area("Enter a sentence to predict its emotion:", "")
    # if input_text:
    #     with st.spinner('Making a prediction...'):
    #         model, vectorizer = load_model_and_vectorizer()
    #         predicted_emotion = predict_emotion(input_text, model, vectorizer)
    #         st.write(f"The predicted emotion is: **{predicted_emotion}**")
