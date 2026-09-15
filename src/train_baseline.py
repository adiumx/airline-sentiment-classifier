"""
Entrena y evalúa dos modelos baseline para clasificación de sentimiento
de tweets de aerolíneas, logueando ambos como runs separados en MLflow:
  1. Dummy classifier (siempre predice la clase mayoritaria) - referencia mínima
  2. TF-IDF + Logistic Regression - modelo clásico entrenado en el dominio

Uso:
    python src/train_baseline.py
"""

import sys
import os
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import load_dataset

DATA_PATH = "Tweets.csv"
EXPERIMENT_NAME = "airline_sentiment_classification"

# Mismo criterio que en el proyecto de tickets: SQLite como backend local,
# ya que el backend de filesystem está en modo mantenimiento en MLflow reciente.
mlflow.set_tracking_uri("sqlite:///mlflow.db")


def _log_run(run_name, model, X_train, y_train, X_test, y_test, params, sklearn_model=True):
    with mlflow.start_run(run_name=run_name):
        for key, value in params.items():
            mlflow.log_param(key, value)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        accuracy = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average="macro")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)

        report = classification_report(y_test, preds)
        report_path = f"classification_report_{run_name}.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)
        os.remove(report_path)

        if sklearn_model:
            mlflow.sklearn.log_model(model, name="model")

        print(f"\n--- {run_name} ---")
        print(f"Accuracy: {accuracy:.3f} | F1 macro: {f1_macro:.3f}")
        print(report)

        return accuracy, f1_macro


def train():
    train_df, test_df = load_dataset(DATA_PATH)

    X_train = train_df["text_clean_tfidf"]
    X_test = test_df["text_clean_tfidf"]
    y_train = train_df["airline_sentiment"]
    y_test = test_df["airline_sentiment"]

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Run 1: Dummy classifier (referencia mínima)
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    _log_run(
        "dummy_baseline", dummy, X_train, y_train, X_test, y_test,
        params={"strategy": "most_frequent"},
        sklearn_model=True,
    )

    # Run 2: TF-IDF + Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    _log_run(
        "tfidf_logreg", pipeline, X_train, y_train, X_test, y_test,
        params={"max_features": 5000, "class_weight": "balanced", "model_type": "TF-IDF + LogisticRegression"},
        sklearn_model=True,
    )


if __name__ == "__main__":
    train()
