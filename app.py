from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Setup
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text preprocessing (same as used in training)
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        subject = request.form.get("subject", "")
        message = request.form.get("message", "")
        full_text = subject + " " + message
        clean_text = preprocess(full_text)
        vector = vectorizer.transform([clean_text])
        prediction = model.predict(vector)[0]
        result = "SPAM" if prediction == 1 else "NOT SPAM"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
