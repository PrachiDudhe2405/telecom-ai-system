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
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "raw" / "twcs.csv"
MODEL_FILE = BASE_DIR / "models" / "social_model.pkl"
CACHE_FILE = BASE_DIR / "data" / "processed" / "tweet_labels_cache.csv"

ROBERTA_THRESHOLD = 0.65
ZERO_SHOT_THRESHOLD = 0.55

CANDIDATE_LABELS = [
    "customer complaint about telecommunications service",
    "general conversation or positive feedback",
]


def clean_text(text: str) -> str:
    """Normalize tweet text for model input."""
    text = str(text).lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def label_with_roberta(texts: list) -> list:
    """Score tweets with cardiffnlp/twitter-roberta-base-sentiment.

    Returns the LABEL_0 (negative sentiment) probability for each tweet.
    LABEL_0 = negative, LABEL_1 = neutral, LABEL_2 = positive.
    top_k=None returns all three scores so we can always extract LABEL_0
    even when it is not the argmax.
    """
    from transformers import pipeline as hf_pipeline

    print("Loading cardiffnlp/twitter-roberta-base-sentiment …")
    pipe = hf_pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        truncation=True,
        max_length=512,
        top_k=None,
    )

    batch_size = 32
    scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc="RoBERTa scoring"):
        batch = texts[i : i + batch_size]
        results = pipe(batch)
        for label_scores in results:
            neg = next(
                (r["score"] for r in label_scores if r["label"] == "LABEL_0"), 0.0
            )
            scores.append(neg)

    return scores


def label_with_zero_shot(texts: list) -> list:
    """Score tweets with facebook/bart-large-mnli zero-shot classification.

    Returns the probability assigned to the complaint candidate label.
    This model is large (~1.6 GB) and slow on CPU — first run takes
    several minutes; results are cached to avoid re-running.
    """
    from transformers import pipeline as hf_pipeline

    print("Loading facebook/bart-large-mnli …")
    pipe = hf_pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        truncation=True,
    )

    scores = []
    for text in tqdm(texts, desc="Zero-shot scoring"):
        if not text.strip():
            scores.append(0.0)
            continue
        result = pipe(text, candidate_labels=CANDIDATE_LABELS)
        complaint_idx = result["labels"].index(CANDIDATE_LABELS[0])
        scores.append(result["scores"][complaint_idx])

    return scores


def create_combined_labels(df: pd.DataFrame, skip_cache: bool = False) -> pd.DataFrame:
    """Apply RoBERTa + zero-shot scoring and build the final complaint_status label.

    Loads from CACHE_FILE if it exists so expensive inference only runs once.
    Label rule: complaint_status = 1 iff
        roberta_negative  > ROBERTA_THRESHOLD  (0.65)  AND
        zero_shot_complaint > ZERO_SHOT_THRESHOLD (0.55)
    """
    texts = df["clean_text"].tolist()

    if not skip_cache and CACHE_FILE.exists():
        print(f"Loading cached labels from {CACHE_FILE}")
        cache = pd.read_csv(CACHE_FILE, index_col=0)
        df = df.copy()
        df["roberta_negative"] = cache["roberta_negative"].values
        df["zero_shot_complaint"] = cache["zero_shot_complaint"].values
    else:
        print("Cache not found — running inference (this may take several minutes) …")
        roberta_scores = label_with_roberta(texts)
        zero_shot_scores = label_with_zero_shot(texts)

        df = df.copy()
        df["roberta_negative"] = roberta_scores
        df["zero_shot_complaint"] = zero_shot_scores

        if not skip_cache:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            cache_df = pd.DataFrame(
                {
                    "roberta_negative": roberta_scores,
                    "zero_shot_complaint": zero_shot_scores,
                },
                index=df.index,
            )
            cache_df.to_csv(CACHE_FILE)
            print(f"Scores cached to {CACHE_FILE}")

    # OR branch catches factual complaints with no strong negative emotion
    # (e.g. "verizon is down, when's the fix?") where RoBERTa scores neutral
    # but zero-shot is highly confident it's a service complaint.
    df["complaint_status"] = (
        (
            (df["roberta_negative"] > ROBERTA_THRESHOLD)
            & (df["zero_shot_complaint"] > ZERO_SHOT_THRESHOLD)
        )
        | (df["zero_shot_complaint"] > 0.85)
    ).astype(int)

    print("\ncomplaint_status value counts (combined RoBERTa + zero-shot labels):")
    print(df["complaint_status"].value_counts())

    print("\n5 example tweets with scores and final label:")
    sample = df[["clean_text", "roberta_negative", "zero_shot_complaint", "complaint_status"]].sample(
        5, random_state=42
    )
    for _, row in sample.iterrows():
        print(
            f"  [{row['complaint_status']}] roberta={row['roberta_negative']:.3f} "
            f"zero_shot={row['zero_shot_complaint']:.3f}  \"{row['clean_text'][:80]}\""
        )

    return df


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
    print(f"Raw text:       {raw_text}")
    print(f"Cleaned text:   {cleaned_text}")
    print(f"Predicted label: {predicted_label} ({label_name})")
    print("Predicted probabilities:")
    for class_label, probability in zip(model.classes_, predicted_probabilities):
        class_name = "complaint" if int(class_label) == 1 else "no complaint"
        print(f"  Class {class_label} ({class_name}): {probability:.4f}")


def main(test_mode: bool = False) -> None:
    nrows = 500 if test_mode else 50000
    if test_mode:
        print("--- TEST MODE: 500 rows, cache skipped ---")
    skip_cache = test_mode
    df = pd.read_csv(DATA_FILE, nrows=nrows)
    print(f"Original shape: {df.shape}")

    # inbound=True means the tweet was sent by a customer to the company.
    filtered_df = df[df["inbound"] == True].copy()
    filtered_df["clean_text"] = filtered_df["text"].fillna("").apply(clean_text)
    print(f"Filtered shape (inbound=True): {filtered_df.shape}")

    filtered_df = create_combined_labels(filtered_df, skip_cache=skip_cache)

    X = filtered_df["clean_text"]
    y = filtered_df["complaint_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTrain size: {X_train.shape[0]}  Test size: {X_test.shape[0]}")
    print("complaint_status distribution in train set:")
    print(y_train.value_counts())

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
    top_words = feature_names[coefficients.argsort()[-10:][::-1]]

    print("\nTop 10 complaint words:")
    print(", ".join(top_words))

    print("\nClassification report:")
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
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", help="Predict one tweet with the saved model.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run on 500 rows without cache to verify labelling quality before the full run.",
    )
    args = parser.parse_args()

    if args.predict:
        predict_single_tweet(args.predict)
    else:
        main(test_mode=args.test)
