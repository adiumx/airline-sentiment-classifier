"""
App de Gradio para el Space de Hugging Face.

Clasifica el sentimiento de un tweet dirigido a una aerolínea usando el
modelo RoBERTa fine-tuned con el dataset de Twitter US Airline Sentiment.
"""
import spaces
import re
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Reemplaza esto por tu propio modelo subido al Hub (ver README para el paso
# de push_to_hub previo a desplegar esta app).
MODEL_ID = "cesarrf/roberta-airline-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_EMOJI = {"negative": "😠", "neutral": "😐", "positive": "😊"}


def clean_for_transformer(text: str) -> str:
    """Misma limpieza mínima usada en entrenamiento (ver src/preprocess.py)."""
    text = re.sub(r'@\w+', '@user', text)
    text = re.sub(r'http\S+', 'http', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@spaces.GPU
def predict_sentiment(text: str):
    if not text or not text.strip():
        return {"negative": 0.0, "neutral": 0.0, "positive": 0.0}

    cleaned = clean_for_transformer(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]

    return {ID2LABEL[i]: float(probs[i]) for i in range(3)}


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        label="Texto del tweet",
        placeholder="ej: @united my flight was delayed 3 hours, terrible service",
        lines=3,
    ),
    outputs=gr.Label(label="Sentimiento", num_top_classes=3),
    title="Clasificador de sentimiento — tweets de aerolíneas",
    description=(
        "RoBERTa fine-tuned sobre el dataset Twitter US Airline Sentiment. "
        "Compara predicciones contra un modelo clásico (TF-IDF + Logistic Regression) "
        "y contra el mismo RoBERTa sin fine-tuning — ver el repo para el análisis completo."
    ),
    examples=[
        ["@united my flight was delayed 3 hours and nobody explained why, terrible service"],
        ["@JetBlue what time does boarding start for flight 123?"],
        ["@Delta thanks for the great customer service, really appreciated it!"],
    ],
)

if __name__ == "__main__":
    demo.launch()
