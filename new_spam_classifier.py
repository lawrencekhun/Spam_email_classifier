"""
Spam Classifier using Enron Dataset
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Preprocessing function
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Load dataset
df = pd.read_csv('enron_spam_data.csv')  # or whatever your exact file name is

# Combine subject + message into one column
df['text'] = df['Subject'].fillna('') + ' ' + df['Message'].fillna('')
df['text'] = df['text'].apply(preprocess)

# Convert labels to binary
df['label'] = df['Spam/Ham'].map({'ham': 0, 'spam': 1})

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Save model and vectorizer for Flask
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
