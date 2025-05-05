# 📧 Spam Email Classifier (Enron Dataset + Flask)

This project is a web-based spam email classifier that uses the **Enron Email Dataset** and a **Naive Bayes algorithm** to predict whether an email is spam or not. It includes a Flask-powered web interface where users can input an email subject and message to receive an instant prediction.

---

## ✅ Features

- Trained on **real-world emails** from the Enron spam dataset
- Uses both the **subject and message** for improved accuracy
- Simple and responsive **Flask web interface**
- Outputs prediction: **SPAM** or **NOT SPAM**
- Preprocessing includes:
  - Lowercasing
  - Stop word removal
  - Stemming
  - TF-IDF vectorization

---

## 🧠 Model Details

- **Algorithm**: Multinomial Naive Bayes
- **Vectorization**: TF-IDF
- **Dataset**: [Enron Spam Dataset](https://www.kaggle.com/datasets/wanderfj/enron-spam)

---

## 🚀 How to Run Locally

### 1. Clone the repo or download the files
Make sure you have:
- `app.py`
- `templates/index.html`
- `model.pkl` and `vectorizer.pkl` (from training script)

### 2. Set up a virtual environment
```bash
python3 -m venv spam-env
source spam-env/bin/activate

### 3. Install dependencies
pip install flask pandas scikit-learn nltk joblib

### 4. Train the model (if needed)
Run new_spam_classifier.py to:
- Load and clean the data
- Train the model
- Save model.pkl and vectorizer.pkl
python3 new_spam_classifier.py

### 5. Start the Flask app
python3 app.py

Then visit http://127.0.0.1:5000 in your browser.

########
📩 Example: SPAM Email
Subject:🚨 Claim Your FREE iPhone 15 Now – Only 2 Left!

Message:
Hi there,

You've been chosen to receive a brand new iPhone 15 – absolutely free!

✅ No credit card required  
✅ Ships in 24 hours  
✅ Offer expires soon!

👉 Click here to claim your reward: http://getiphone-now.scam-site.biz

Act fast — this is your last chance!

- The Rewards Team
Expected Prediction: SPAM
########


########
📬 Example: NOT SPAM Email
Subject: Project Update: Marketing Campaign Launch Plan

Message:
Hey team,

Just a quick update on the marketing campaign timeline. We’re set to launch the first round of social media ads on Monday morning. All creative assets have been finalized and approved.

Please check the shared drive for the full rollout plan, and let me know if anything’s missing.

Thanks,  
Jessica
Expected Prediction: NOT SPAM
########