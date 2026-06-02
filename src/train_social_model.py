import argparse
from pathlib import Path
import re
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "raw" / "twcs.csv"
MODEL_FILE = BASE_DIR / "models" / "social_model.pkl"

VADER_COMPLAINT_THRESHOLD = -0.2


def clean_text(text: str) -> str:
    """Normalize tweet text for simple complaint analysis."""
    text = str(text).lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def predict_single_tweet(raw_text: str) -> None:
    """Load the saved model and predict one tweet."""
    if not MODEL_FILE.exists():
        print("Saved model not found. Run training first with: python src/train_social_model.py")
        sys.exit(1)

    model = joblib.load(MODEL_FILE)
    cleaned_text = clean_text(raw_text)
    predicted_label = int(model.predict([cleaned_text])[0])
    predicted_probabilities = model.predict_proba([cleaned_text])[0]

    label_name = "complaint" if predicted_label == 1 else "no complaint"

    print("\nYour tweet prediction:")
    print(f"Raw text: {raw_text}")
    print(f"Cleaned text: {cleaned_text}")
    print(f"Predicted label: {predicted_label} ({label_name})")
    print("Predicted probabilities:")
    for class_label, probability in zip(model.classes_, predicted_probabilities):
        class_name = "complaint" if int(class_label) == 1 else "no complaint"
        print(f"Class {class_label} ({class_name}): {probability:.4f}")


def main() -> None:
    df = pd.read_csv(DATA_FILE, nrows=50000)
    print(f"Original shape: {df.shape}")

    # inbound=True means the tweet was sent by a customer to the company,
    # rather than being a support-agent/company reply.
    filtered_df = df[df["inbound"] == True].copy()
    filtered_df["clean_text"] = filtered_df["text"].fillna("").apply(clean_text)

    analyzer = SentimentIntensityAnalyzer()
    filtered_df["complaint_status"] = (
        filtered_df["clean_text"]
        .apply(lambda t: analyzer.polarity_scores(t)["compound"])
        .lt(VADER_COMPLAINT_THRESHOLD)
        .astype(int)
    )

    X = filtered_df["clean_text"]
    y = filtered_df["complaint_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(max_features=1000, stop_words="english"),
            ),
            (
                "logreg",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    feature_names = model.named_steps["tfidf"].get_feature_names_out()
    coefficients = model.named_steps["logreg"].coef_[0]
    top_indices = coefficients.argsort()[-10:][::-1]
    top_words = feature_names[top_indices]
    accuracy = accuracy_score(y_test, y_pred)
    sample_text = "My internet is not working and calls keep dropping"
    cleaned_sample_text = clean_text(sample_text)
    sample_prediction = model.predict([cleaned_sample_text])[0]
    sample_probabilities = model.predict_proba([cleaned_sample_text])[0]

    print(f"Filtered shape (inbound=True only): {filtered_df.shape}")
    print("\nFirst 3 rows of filtered data:")
    print(filtered_df.head(3))
    print("\nOriginal text and clean_text:")
    print(filtered_df[["text", "clean_text"]].head(5))
    print("\ncomplaint_status value counts:")
    print(filtered_df["complaint_status"].value_counts())
    print("\nTrain/test sizes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    print("\ncomplaint_status distribution in train set:")
    print(y_train.value_counts())
    print("\ncomplaint_status distribution in test set:")
    print(y_test.value_counts())
    print("\nModel trained successfully")
    print("Top complaint words:")
    print(", ".join(top_words))
    print("\nEvaluation on test set:")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification report:")
    # Precision for the complaint class: when the model predicts complaint,
    # how often that prediction is actually correct.
    # Recall for the complaint class: out of all real complaint tweets,
    # how many the model successfully finds.
    # F1-score for the complaint class: the balance between precision and recall.
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["no complaint", "complaint"],
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print("\nModel saved to models/social_model.pkl")
    print("\nSample prediction:")
    print(f"Text: {sample_text}")
    print(f"Cleaned text: {cleaned_sample_text}")
    print(f"Predicted label: {sample_prediction}")
    print("Predicted probabilities:")
    for class_label, probability in zip(model.classes_, sample_probabilities):
        print(f"Class {class_label}: {probability:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predict",
        help="Predict one tweet with the saved social model.",
    )
    args = parser.parse_args()

    if args.predict:
        predict_single_tweet(args.predict)
    else:
        main()
