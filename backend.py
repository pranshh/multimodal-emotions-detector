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
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
import joblib
import os

# Download necessary NLTK packages
nltk.download('stopwords')
nltk.download('punkt')

# Preprocessing Functions
def clean_tweet(tweet):
    tweet = re.sub(r"(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", str(tweet))
    tweet = re.sub(r"\s+", " ", tweet).strip()
    return tweet

def nltk_preprocess(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(filtered_words)

def train_and_save_best_model(dataset_path):
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Please check the path.")
        return

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    print("Previewing dataset:")
    print(df.head())

    # Preprocess text data
    print("Preprocessing text data...")
    df['cleaned_tweet'] = df['text'].apply(clean_tweet)
    df['processed_text'] = df['cleaned_tweet'].apply(nltk_preprocess)

    # Vectorize text and combine with intensity
    vectorizer = CountVectorizer()
    X_text = vectorizer.fit_transform(df['processed_text'])
    X_intensity = np.array(df['intensity']).reshape(-1, 1)
    X = hstack((X_text, X_intensity))

    # Map emotions to numeric values
    label_mapping = {'anger': 0, 'fear': 1, 'joy': 2, 'sadness': 3}
    df['numeric_emotion'] = df['emotion'].map(label_mapping)
    y = df['numeric_emotion']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models and select the best
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }

    best_model = None
    best_accuracy = 0

    print("Training models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy * 100:.2f}%")

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    # Save the best model and vectorizer
    if best_model is not None:
        print(f"Best model: {type(best_model).__name__} with accuracy: {best_accuracy * 100:.2f}%")
        joblib.dump(best_model, "best_emotion_model.pkl")
        joblib.dump(vectorizer, "vectorizer.pkl")
        print("Best model and vectorizer saved successfully!")
    else:
        print("No model was trained successfully.")

# Invoke the function
if __name__ == "__main__":
    dataset_path = "wassa-2017.csv" 
    train_and_save_best_model(dataset_path)
