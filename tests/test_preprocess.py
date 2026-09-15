import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.preprocess import clean_for_tfidf, clean_for_transformer, load_dataset


# --- clean_for_tfidf ---

def test_clean_for_tfidf_removes_mentions():
    text = "@united this flight was awful"
    result = clean_for_tfidf(text)
    assert "@united" not in result
    assert "united" not in result  # se quita la mención completa, no solo el @


def test_clean_for_tfidf_removes_urls():
    text = "check this out http://example.com/flight now"
    result = clean_for_tfidf(text)
    assert "http" not in result


def test_clean_for_tfidf_lowercases():
    text = "AWFUL Flight Experience"
    result = clean_for_tfidf(text)
    assert result == result.lower()


def test_clean_for_tfidf_removes_punctuation_and_emoji():
    text = "worst service ever!! 😡 #neverflying"
    result = clean_for_tfidf(text)
    for char in "!#😡":
        assert char not in result


def test_clean_for_tfidf_normalizes_whitespace():
    text = "too   many     spaces"
    result = clean_for_tfidf(text)
    assert "  " not in result
    assert result == result.strip()


def test_clean_for_tfidf_handles_empty_input():
    assert clean_for_tfidf("") == ""


# --- clean_for_transformer ---

def test_clean_for_transformer_normalizes_mentions_to_user_token():
    text = "@united @AmericanAir this is bad"
    result = clean_for_transformer(text)
    assert "@united" not in result
    assert "@AmericanAir" not in result
    assert result.count("@user") == 2


def test_clean_for_transformer_normalizes_urls():
    text = "details here http://example.com/flight123"
    result = clean_for_transformer(text)
    assert "http://example.com/flight123" not in result
    assert "http" in result


def test_clean_for_transformer_preserves_case_and_punctuation():
    # A diferencia de clean_for_tfidf, aquí NO se debe quitar mayúsculas ni
    # puntuación: el transformer fue entrenado esperando ese formato natural.
    text = "This is TERRIBLE!!"
    result = clean_for_transformer(text)
    assert "TERRIBLE" in result
    assert "!!" in result


def test_clean_for_transformer_normalizes_whitespace():
    text = "@united   delayed    again"
    result = clean_for_transformer(text)
    assert "  " not in result


# --- load_dataset ---

def test_load_dataset_returns_train_and_test_splits(tmp_path):
    rows = []
    for i in range(20):
        rows.append({"text": f"@united flight was bad {i}", "airline_sentiment": "negative"})
    for i in range(10):
        rows.append({"text": f"@united when does it board {i}", "airline_sentiment": "neutral"})
    for i in range(10):
        rows.append({"text": f"@united great service {i}", "airline_sentiment": "positive"})

    csv_path = tmp_path / "test_tweets.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    train_df, test_df = load_dataset(str(csv_path), test_size=0.2, random_state=42)

    assert len(train_df) + len(test_df) == 40
    assert len(test_df) == 8  # 20% de 40


def test_load_dataset_adds_both_cleaned_columns(tmp_path):
    rows = [
        {"text": "@united this is BAD!!", "airline_sentiment": "negative"},
        {"text": "@united another bad one", "airline_sentiment": "negative"},
        {"text": "@united great service", "airline_sentiment": "positive"},
        {"text": "@united another great one", "airline_sentiment": "positive"},
    ]
    csv_path = tmp_path / "test_tweets.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    train_df, test_df = load_dataset(str(csv_path), test_size=0.5, random_state=42)
    combined = pd.concat([train_df, test_df])

    assert "text_clean_tfidf" in combined.columns
    assert "text_clean_transformer" in combined.columns


def test_load_dataset_split_is_stratified(tmp_path):
    # 100 negative, 20 neutral, 10 positive: proporciones deliberadamente
    # desbalanceadas, como el dataset real.
    rows = []
    rows += [{"text": f"bad flight {i}", "airline_sentiment": "negative"} for i in range(100)]
    rows += [{"text": f"question {i}", "airline_sentiment": "neutral"} for i in range(20)]
    rows += [{"text": f"great flight {i}", "airline_sentiment": "positive"} for i in range(10)]

    csv_path = tmp_path / "test_tweets.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    train_df, test_df = load_dataset(str(csv_path), test_size=0.2, random_state=42)

    train_prop = train_df["airline_sentiment"].value_counts(normalize=True)
    test_prop = test_df["airline_sentiment"].value_counts(normalize=True)

    # Las proporciones de train y test deben ser muy parecidas (estratificado),
    # con una tolerancia chica para redondeos.
    for label in ["negative", "neutral", "positive"]:
        assert abs(train_prop[label] - test_prop[label]) < 0.05
