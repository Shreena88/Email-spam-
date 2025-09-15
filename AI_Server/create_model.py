import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


DATA_FILE = "Merged1_data.csv"
MODEL_PKL = "better_spam_model.pkl"
TRANSFORMER_DIR = "transformer_model"
BASE_MODEL = "distilbert-base-uncased"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer, max_length: int = 256):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


@dataclass
class TransformerSpamClassifier:
    """
    A lightweight wrapper that exposes an sklearn-like API around a fine-tuned
    Hugging Face transformer. Only the directory path is pickled; weights are
    loaded on demand when used.
    """

    model_dir: str
    label2id: Dict[str, int]

    def __post_init__(self) -> None:
        self.id2label = {v: k for k, v in self.label2id.items()}
        self._tokenizer = None
        self._model = None

    @property
    def classes_(self) -> np.ndarray:
        # Order must match predict_proba columns
        ordered = [self.id2label[idx] for idx in sorted(self.id2label.keys())]
        return np.array(ordered)

    def _ensure_loaded(self) -> None:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self._model.eval()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        self._ensure_loaded()
        if isinstance(texts, str):
            texts = [texts]
        enc = self._tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**enc)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs  # shape: (n_samples, n_classes) ordered by id

    def predict(self, texts: List[str]) -> List[str]:
        probs = self.predict_proba(texts)
        pred_ids = probs.argmax(axis=1)
        return [self.id2label[int(i)] for i in pred_ids]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def load_and_prepare_data(csv_path: str):
    print(f"⏳ Loading data from '{csv_path}' ...")
    df = pd.read_csv(csv_path, on_bad_lines="skip", low_memory=False, encoding="latin-1")
    df.columns = [c.strip().lower() for c in df.columns]

    text_col = next((c for c in ["message", "text", "email", "content", "body"] if c in df.columns), None)
    label_col = next((c for c in ["category", "label", "class", "target", "tag"] if c in df.columns), None)
    if not text_col or not label_col:
        raise ValueError("Could not detect text/label columns in CSV.")

    df = df.dropna(subset=[text_col, label_col])
    df[text_col] = df[text_col].apply(clean_text)
    df[label_col] = (
        df[label_col]
        .astype(str)
        .str.lower()
        .str.strip()
        .replace({
            "phishing": "spam",
            "junk": "spam",
            "advertisement": "spam",
            "ads": "spam",
            "promo": "spam",
            "marketing": "spam",
            "spam": "spam",
            "ham": "ham",
            "not spam": "ham",
            "legit": "ham",
            "normal": "ham",
        })
    )

    df = df[df[label_col].isin(["spam", "ham"])]
    print("Label distribution:\n", df[label_col].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col].tolist(),
        (df[label_col] == "spam").astype(int).tolist(),
        test_size=0.2,
        random_state=42,
        stratify=(df[label_col] == "spam").astype(int).tolist(),
    )

    return X_train, X_test, y_train, y_test


def train_transformer_and_save():
    X_train, X_test, y_train, y_test = load_and_prepare_data(DATA_FILE)

    label2id = {"ham": 0, "spam": 1}
    id2label = {0: "ham", 1: "spam"}

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    train_dataset = TextDataset(X_train, y_train, tokenizer)
    eval_dataset = TextDataset(X_test, y_test, tokenizer)

    training_args = TrainingArguments(
        output_dir="./model_out",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save transformer in a reusable directory
    os.makedirs(TRANSFORMER_DIR, exist_ok=True)
    trainer.save_model(TRANSFORMER_DIR)
    tokenizer.save_pretrained(TRANSFORMER_DIR)

    # Create a lightweight, pickleable wrapper
    wrapper = TransformerSpamClassifier(model_dir=TRANSFORMER_DIR, label2id=label2id)
    joblib.dump(wrapper, MODEL_PKL)
    print(f"✅ Saved wrapper to '{MODEL_PKL}' and model to '{TRANSFORMER_DIR}'")

    # Quick sanity check
    samples = [
        "Congratulations! You won a free iPhone. Click here to claim.",
        "Please review the attached project report before our meeting.",
        "Your bank account has been suspended. Login here to verify.",
        "Join us for dinner at 7pm tonight.",
    ]
    preds = wrapper.predict(samples)
    for s, p in zip(samples, preds):
        print(f"- {p.upper()} :: {s}")

    return wrapper


if __name__ == "__main__":
    if not os.path.exists(MODEL_PKL):
        train_transformer_and_save()
    else:
        print(f"🔍 Found existing '{MODEL_PKL}'. If you want to retrain, delete it and rerun.")
