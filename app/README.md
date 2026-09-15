---
title: Airline Sentiment Classifier
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# Clasificador de Sentimiento — Tweets de Aerolíneas

Demo de un modelo RoBERTa fine-tuned sobre el dataset [Twitter US Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment), que clasifica un tweet dirigido a una aerolínea como negativo, neutral o positivo.

Este modelo forma parte de una comparación de 4 enfoques (baseline ingenuo, TF-IDF + Logistic Regression, RoBERTa zero-shot, RoBERTa fine-tuned) documentada en el [repositorio del proyecto](#) — el fine-tuning fue el que mejor desempeño obtuvo (F1 macro: 0.816).
