"""
Preprocesamiento de texto para tweets de sentimiento de aerolíneas.
Dos funciones de limpieza distintas: una agresiva para TF-IDF,
otra mínima para el transformer (ver README para la justificación).
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def clean_for_tfidf(text: str) -> str:
    # TODO: pega aquí tu función exacta del notebook
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_for_transformer(text: str) -> str:
    # TODO: pega aquí tu función exacta del notebook
    text = re.sub(r'@\w+', '@user', text)
    text = re.sub(r'http\S+', 'http', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_dataset(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """Carga el CSV, limpia el texto y devuelve train_df, test_df ya divididos."""
    df = pd.read_csv(csv_path)
    df_model = df[['text', 'airline_sentiment']].copy()
    df_model['text_clean_tfidf'] = df_model['text'].apply(clean_for_tfidf)
    df_model['text_clean_transformer'] = df_model['text'].apply(clean_for_transformer)

    train_idx, test_idx = train_test_split(
        df_model.index,
        test_size=test_size,
        random_state=random_state,
        stratify=df_model['airline_sentiment']
    )

    train_df = df_model.loc[train_idx]
    test_df = df_model.loc[test_idx]
    return train_df, test_df