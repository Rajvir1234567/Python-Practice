
import sys
import re
import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import joblib


# ---------------------------------------------------------------------------
# 1. Built-in sample dataset (used only if no CSV is supplied)
#    A tiny illustrative set; swap in a real dataset like the SMS Spam
#    Collection (5,574 messages) for production-quality results.
# ---------------------------------------------------------------------------
SAMPLE_DATA = [
    ("spam", "WINNER!! You have been selected to receive a $1000 cash prize! Call now to claim!!!"),
    ("spam", "URGENT: Your account has been suspended. Click here to verify your details immediately."),
    ("spam", "Congratulations! You've won a free iPhone. Click this link to claim your prize now."),
    ("spam", "Get rich quick! Invest $100 today and earn $10,000 in a week. Limited time offer!"),
    ("spam", "FREE entry into our $5000 weekly draw just by texting WIN to 80086 now!!!"),
    ("spam", "You have 1 new voicemail. To listen dial this premium number: 090-xxx-xxxx"),
    ("spam", "Hot singles in your area want to meet you tonight! Click here now."),
    ("spam", "Your loan has been approved for $50000. No credit check needed. Reply YES now."),
    ("spam", "Claim your free gift card worth $500 by clicking the link below before it expires."),
    ("spam", "CONGRATULATIONS! Your number has won a lottery of £1,000,000. Contact claims agent now."),
    ("spam", "Limited offer: buy one get one free on all products. Shop now before stock runs out!"),
    ("spam", "Act now! Your subscription is about to expire, renew immediately to avoid charges."),
    ("spam", "Nigerian prince needs your help to transfer $10 million, you will get 20% commission."),
    ("spam", "Text STOP to unsubscribe or continue to receive amazing daily deals just for you!"),
    ("spam", "Free trial! No credit card needed! Sign up now and get instant access to premium content."),
    ("spam", "Your PayPal account has unusual activity. Verify your identity now to avoid suspension."),
    ("spam", "You are pre-approved for a $25000 loan. Apply now, no fees, instant approval guaranteed."),
    ("spam", "Cheap meds online, no prescription needed, discreet shipping, order now and save 80%."),
    ("spam", "Work from home and earn $5000 a week! No experience required, sign up today!"),
    ("spam", "Click here to reset your bank password immediately or your account will be locked."),
    ("ham", "Hey, are we still meeting for lunch tomorrow at noon?"),
    ("ham", "Can you send me the report before end of day? Thanks!"),
    ("ham", "Happy birthday! Hope you have a wonderful day with family and friends."),
    ("ham", "I'll be running about 10 minutes late for the meeting, sorry about that."),
    ("ham", "Don't forget to pick up milk and eggs on your way home."),
    ("ham", "The movie was great, we should watch the sequel next weekend."),
    ("ham", "Thanks for helping me move last weekend, I really appreciate it."),
    ("ham", "Let's catch up over coffee sometime this week if you're free."),
    ("ham", "The project deadline has been moved to next Friday, please plan accordingly."),
    ("ham", "Mom called, she wants us to come over for dinner on Sunday."),
    ("ham", "Great job on the presentation today, the client seemed really impressed."),
    ("ham", "Can you review my code before I push it? Just a couple of small functions."),
    ("ham", "I finished reading that book you recommended, it was really good."),
    ("ham", "Reminder: dentist appointment at 3pm tomorrow, don't forget."),
    ("ham", "The weather looks nice this weekend, want to go hiking?"),
    ("ham", "Sorry I missed your call earlier, was in a meeting. What's up?"),
    ("ham", "Please find attached the invoice for last month's services."),
    ("ham", "Our flight got delayed by two hours, we'll land around 9pm now."),
    ("ham", "I added the notes from today's standup to the shared doc."),
    ("ham", "Congrats on the new job! You totally deserve it, well done."),
]


def load_data(csv_path=None):
    """Load dataset from CSV (columns: label, text) or fall back to sample data."""
    if csv_path:
        df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in df.columns}
        if "label" not in cols or "text" not in cols:
            raise ValueError("CSV must contain 'label' and 'text' columns")
        df = df.rename(columns={cols["label"]: "label", cols["text"]: "text"})
    else:
        df = pd.DataFrame(SAMPLE_DATA, columns=["label", "text"])

    # Normalize labels to 0 (ham) / 1 (spam)
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["label"] = df["label"].map(
        lambda x: 1 if x in ("spam", "1") else 0
    )
    df = df.dropna(subset=["text", "label"])
    return df


def clean_text(text):
    """Basic text normalization: lowercase, strip URLs/numbers/punctuation."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"\S+@\S+", " ", text)                    # emails
    text = re.sub(r"\d+", " ", text)                        # numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_pipeline_data(df):
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)
    return df


def train_model(df, test_size=0.2, random_state=42):
    X = df["clean_text"]
    y = df["label"]

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",   # handles spam/ham imbalance
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X_train_vec, y_train)

    return model, vectorizer, X_train_vec, X_test_vec, y_train, y_test


def evaluate(model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)[:, 1]

    print("=" * 55)
    print("MODEL EVALUATION")
    print("=" * 55)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    if y_test.nunique() > 1:
        print(f"ROC AUC  : {roc_auc_score(y_test, y_proba):.4f}")

    print("\nConfusion Matrix (rows=actual, cols=predicted) [ham, spam]:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"], zero_division=0))


def top_spam_features(model, vectorizer, n=15):
    """Show the words/phrases most strongly associated with spam."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]
    top_idx = np.argsort(coefs)[-n:][::-1]
    print(f"\nTop {n} words/phrases indicating SPAM:")
    for i in top_idx:
        print(f"  {feature_names[i]:<20} weight={coefs[i]:.3f}")


def predict_messages(model, vectorizer, messages):
    cleaned = [clean_text(m) for m in messages]
    vec = vectorizer.transform(cleaned)
    preds = model.predict(vec)
    probs = model.predict_proba(vec)[:, 1]
    print("\nPredictions on new messages:")
    for msg, pred, prob in zip(messages, preds, probs):
        label = "SPAM" if pred == 1 else "HAM"
        print(f"  [{label:4}] (spam prob={prob:.3f})  {msg}")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("Loading data..." + (f" ({csv_path})" if csv_path else " (built-in sample dataset)"))
    df = load_data(csv_path)
    df = build_pipeline_data(df)
    print(f"Loaded {len(df)} messages | spam={df['label'].sum()} ham={(df['label']==0).sum()}")

    model, vectorizer, X_train_vec, X_test_vec, y_train, y_test = train_model(df)

    evaluate(model, X_test_vec, y_test)
    top_spam_features(model, vectorizer)

    # 5-fold cross-validation on the full TF-IDF matrix for a more robust estimate
    try:
        full_vec = vectorizer.transform(df["clean_text"])
        cv_scores = cross_val_score(model, full_vec, df["label"], cv=5, scoring="f1")
        print(f"\n5-fold CV F1 scores: {np.round(cv_scores, 3)}")
        print(f"Mean CV F1: {cv_scores.mean():.4f}")
    except ValueError:
        print("\n(Skipping cross-validation: not enough samples per class)")

    # Demo predictions on unseen messages
    demo_messages = [
        "Congratulations! You've won a free vacation, click here to claim now!!!",
        "Hey, can we reschedule our meeting to 3pm tomorrow?",
        "URGENT: verify your bank account now or it will be suspended",
        "Thanks for dinner last night, let's do it again soon.",
    ]
    predict_messages(model, vectorizer, demo_messages)

    # Persist model + vectorizer for reuse
    joblib.dump(model, "spam_model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")
    print("\nSaved trained model to 'spam_model.joblib' and 'vectorizer.joblib'")


if __name__ == "__main__":
    main()