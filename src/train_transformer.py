"""
Evalúa RoBERTa zero-shot y hace fine-tuning sobre el dominio de aerolíneas,
logueando ambos como runs separados en el mismo experimento de MLflow que
train_baseline.py, para poder comparar los 4 enfoques lado a lado.

Uso:
    python src/train_transformer.py            # corre zero-shot + fine-tuning
    python src/train_transformer.py --skip-zeroshot   # solo fine-tuning
"""

import sys
import os
import argparse
import numpy as np
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow.transformers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import load_dataset, LABEL2ID, ID2LABEL

DATA_PATH = "Tweets.csv"
EXPERIMENT_NAME = "airline_sentiment_classification"
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
# El modelo pre-entrenado usa su propio orden de labels (LABEL_0/1/2), que no
# necesariamente coincide con nuestro LABEL2ID -> se mapea explícitamente.
PRETRAINED_LABEL_MAP = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

mlflow.set_tracking_uri("sqlite:///mlflow.db")


def _log_metrics_and_report(run_name, y_test, preds, params):
    accuracy = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro")

    for key, value in params.items():
        mlflow.log_param(key, value)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_macro", f1_macro)

    report = classification_report(y_test, preds)
    report_path = f"classification_report_{run_name}.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)
    os.remove(report_path)

    print(f"\n--- {run_name} ---")
    print(f"Accuracy: {accuracy:.3f} | F1 macro: {f1_macro:.3f}")
    print(report)
    return accuracy, f1_macro


def run_zeroshot(test_df):
    from transformers import pipeline

    X_test = test_df["text_clean_transformer"]
    y_test = test_df["airline_sentiment"]

    device = 0  # GPU; usar -1 si no hay GPU disponible
    sentiment_model = pipeline("sentiment-analysis", model=MODEL_NAME, device=device)

    with mlflow.start_run(run_name="roberta_zeroshot"):
        resultados = sentiment_model(list(X_test), batch_size=64, truncation=True)
        preds = [PRETRAINED_LABEL_MAP[r["label"]] for r in resultados]

        _log_metrics_and_report(
            "roberta_zeroshot", y_test, preds,
            params={"model_type": MODEL_NAME, "fine_tuned": False},
        )


def run_finetune(train_df, test_df, num_epochs=3, batch_size=16):
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    train_dataset = Dataset.from_dict({
        "text": train_df["text_clean_transformer"].tolist(),
        "label": train_df["airline_sentiment"].map(LABEL2ID).tolist(),
    })
    test_dataset = Dataset.from_dict({
        "text": test_df["text_clean_transformer"].tolist(),
        "label": test_df["airline_sentiment"].map(LABEL2ID).tolist(),
    })

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    training_args = TrainingArguments(
        output_dir="./roberta_finetuned",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        report_to="mlflow",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run(run_name="roberta_finetuned"):
        mlflow.log_param("model_type", MODEL_NAME)
        mlflow.log_param("fine_tuned", True)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)

        trainer.train()

        predictions = trainer.predict(test_dataset)
        preds_ids = np.argmax(predictions.predictions, axis=1)
        preds = [ID2LABEL[int(p)] for p in preds_ids]
        y_test = test_df["airline_sentiment"]

        accuracy = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average="macro")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)

        report = classification_report(y_test, preds)
        report_path = "classification_report_roberta_finetuned.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)
        os.remove(report_path)

        print(f"\n--- roberta_finetuned ---")
        print(f"Accuracy: {accuracy:.3f} | F1 macro: {f1_macro:.3f}")
        print(report)

        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            name="model",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-zeroshot", action="store_true")
    args = parser.parse_args()

    train_df, test_df = load_dataset(DATA_PATH)
    mlflow.set_experiment(EXPERIMENT_NAME)

    if not args.skip_zeroshot:
        run_zeroshot(test_df)

    run_finetune(train_df, test_df)
