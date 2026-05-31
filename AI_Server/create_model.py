import pandas as pd
import re
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Resolve paths relative to the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "Merged1_data.csv")
MODEL_FILE = os.path.join(SCRIPT_DIR, "better_spam_model.pkl")

def clean_text(text):
    """
    Improved text cleaning:
    - Replace URLs with 'urlplaceholder'
    - Replace emails with 'emailplaceholder'
    - Replace numbers with 'numberplaceholder'
    - Lowercase and remove punctuation (replacing with spaces to avoid merging words)
    - Remove extra spaces
    """
    text = str(text).lower()
    
    # 1. Replace URLs
    text = re.sub(r"https?://\S+|www\.\S+", " urlplaceholder ", text)
    
    # 2. Replace email addresses
    text = re.sub(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", " emailplaceholder ", text)
    
    # 3. Replace numbers
    text = re.sub(r"\b\d+(?:[.,]\d+)*\b", " numberplaceholder ", text)
    
    # 4. Remove all non-alphabetic/non-whitespace characters (replace with space to avoid word merging)
    text = re.sub(r"[^a-z\s]", " ", text)
    
    # 5. Clean up extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def train_and_save_model():
    print(f"Loading data from '{DATA_FILE}' ...")
    data = pd.read_csv(DATA_FILE, on_bad_lines='skip', low_memory=False, encoding='latin-1')

    # normalize column names
    data.columns = [c.strip().lower() for c in data.columns]

    # detect columns
    text_col = next((c for c in ['message','text','email','content','body'] if c in data.columns), None)
    label_col = next((c for c in ['category','label','class','target','tag'] if c in data.columns), None)

    if not text_col or not label_col:
        raise ValueError("Could not detect text/label columns in CSV.")

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

    # train/test split (split the original dataset first)
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data[label_col]
    )

    X_train_raw = train_data[text_col]
    y_train = train_data[label_col]
    X_test = test_data[text_col]
    y_test = test_data[label_col]

    print("\nUnbalanced train label distribution:")
    print(y_train.value_counts())

    # build training pipeline with SMOTE
    training_pipeline = ImbPipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(1,2),     # unigrams + bigrams
            max_df=0.95,           # drop overly common words
            min_df=3               # ignore rare words
        )),
        ('smote', SMOTE(random_state=42)),
        ('nb', MultinomialNB(alpha=0.3, fit_prior=True))
    ])

    # train pipeline (SMOTE is applied to TF-IDF features)
    print("\nTraining model with SMOTE balancing...")
    training_pipeline.fit(X_train_raw, y_train)

    # Reconstruct standard scikit-learn pipeline for inference (removes SMOTE dependency during prediction/loading)
    model = SkPipeline([
        ('tfidf', training_pipeline.named_steps['tfidf']),
        ('nb', training_pipeline.named_steps['nb'])
    ])

    # evaluate
    y_pred = model.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred, digits=4))
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Overall Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # save
    joblib.dump(model, MODEL_FILE)
    print(f"\nModel saved as '{MODEL_FILE}'")

    return model

if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE):
        model = train_and_save_model()
    else:
        print(f"Loading existing model '{MODEL_FILE}'...")
        model = joblib.load(MODEL_FILE)
        print("Model loaded successfully.")

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
