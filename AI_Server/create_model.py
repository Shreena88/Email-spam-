import pandas as pd
import re
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

DATA_FILE = "Merged1_data.csv"
MODEL_FILE = "better_spam_model.pkl"

def clean_text(text):
    """
    Basic text cleaning: lowercase, remove numbers/punctuation/extra spaces.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def train_and_save_model():
    print(f"⏳ Loading data from '{DATA_FILE}' ...")
    data = pd.read_csv(DATA_FILE, on_bad_lines='skip', low_memory=False, encoding='latin-1')

    # normalize column names
    data.columns = [c.strip().lower() for c in data.columns]

    # detect columns
    text_col = next((c for c in ['message','text','email','content','body'] if c in data.columns), None)
    label_col = next((c for c in ['category','label','class','target','tag'] if c in data.columns), None)

    if not text_col or not label_col:
        raise ValueError("❌ Could not detect text/label columns in CSV.")

    # drop missing
    data = data.dropna(subset=[text_col, label_col])

    # normalize labels
    data[label_col] = data[label_col].astype(str).str.lower().str.strip().replace({
        'phishing':'spam','junk':'spam','advertisement':'spam','ads':'spam',
        'promo':'spam','marketing':'spam','spam':'spam',
        'ham':'ham','not spam':'ham','legit':'ham','normal':'ham'
    })

    # keep only spam/ham
    data = data[data[label_col].isin(['spam','ham'])]

    # clean text
    data[text_col] = data[text_col].apply(clean_text)

    print("\nLabel distribution:")
    print(data[label_col].value_counts())

    # balance dataset
    spam = data[data[label_col]=='spam']
    ham = data[data[label_col]=='ham']
    if len(spam) < len(ham):
        ham_down = resample(ham, replace=False, n_samples=len(spam), random_state=42)
        data_bal = pd.concat([spam, ham_down])
    else:
        spam_up = resample(spam, replace=True, n_samples=len(ham), random_state=42)
        data_bal = pd.concat([ham, spam_up])

    print("\nBalanced label distribution:")
    print(data_bal[label_col].value_counts())

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        data_bal[text_col], data_bal[label_col], test_size=0.2, random_state=42, stratify=data_bal[label_col]
    )

    # build pipeline
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(1,2),     # unigrams + bigrams
            max_df=0.95,           # drop overly common words
            min_df=3               # ignore rare words
        )),
        ('nb', MultinomialNB(alpha=0.3, fit_prior=True))
    ])

    # train
    model.fit(X_train, y_train)

    # evaluate
    y_pred = model.predict(X_test)
    print("\n📊 Model Performance:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # save
    joblib.dump(model, MODEL_FILE)
    print(f"\n✅ Model saved as '{MODEL_FILE}'")

    return model

if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE):
        model = train_and_save_model()
    else:
        print(f"🔍 Loading existing model '{MODEL_FILE}'...")
        model = joblib.load(MODEL_FILE)
        print("✅ Model loaded successfully.")

    # quick test
    samples = [
        "Congratulations! You won a free iPhone. Click here to claim.",
        "Please review the attached project report before our meeting.",
        "Your bank account has been suspended. Login here to verify.",
        "Join us for dinner at 7pm tonight."
    ]
    preds = model.predict(samples)
    for msg, pred in zip(samples, preds):
        print(f"\n> {msg}\nPrediction: {pred.upper()}")
