# Clasificación de Emociones con NLP: De RNNs a Transformers

Este repositorio contiene un estudio comparativo sobre el uso de diversas arquitecturas de Deep Learning para el Procesamiento de Lenguaje Natural (NLP). El proyecto aborda la clasificación multiclase de sentimientos utilizando el dataset **GoEmotions** (el cual será mapeado en relación a las 7 emociones básicas de Ekman).

## Descripción del Proyecto

El objetivo principal es analizar la evolución de los modelos de lenguaje, comparando modelos tradicionales basados en recurrencia **(GRU)** frente a arquitecturas modernas basadas en **Atención** y **Transformers**. 

A lo largo del proyecto, se resuelven retos críticos como el desbalanceo severo de clases, la ambigüedad semántica en comentarios de redes sociales y la optimización de modelos preentrenados de gran escala.

## Estructura del Repositorio

* **`01_eda_and_glove_embeddings.ipynb`**: Análisis exploratorio (EDA), extracción de palabras clave mediante TF-IDF y visualización espacial de embeddings GloVe con la técnica **t-SNE**.
* **`02_pytorch_gru_baseline.ipynb`**: Implementación en PyTorch de una red recurrente (GRU) unidireccional utilizando `packed padded sequences`.
* **`03_custom_pytorch_transformer.ipynb`**: Diseño y construcción desde cero de un **Transformer** en PyTorch, incluyendo `Multi-Head Attention` y `Global Masked Pooling`.
* **`04_huggingface_bert_finetuning.ipynb`**: Aplicación de Transfer Learning con el modelo `bert-base-uncased`, comparando el Fine-Tuning completo frente al entrenamiento selectivo de capas superiores.

## Tecnologías Utilizadas

* **Framework Principal**: PyTorch
* **Ecosistema Transformer**: Hugging Face (`transformers`, `datasets`, Trainer API)
* **NLP & Embeddings**: Gensim (GloVe), Scikit-Learn
* **Visualización**: Seaborn, Matplotlib, t-SNE

## Resultados Comparativos

| Modelo | Exactitud (Accuracy) | F1-Score (Macro) |
| :--- | :---: | :---: |
| **BERT (Fine-Tuning Completo)** | **68.78%** | **0.617** |
| BERT (Capas Superiores + EarlyStop) | 68.45% | 0.609 |
| Transformer (Desde cero) | 59.40% | 0.519 |
| GRU (Baseline) | 59.00% | 0.490 |
| BERT (Base Congelada) | 38.19% | 0.094 |

## Conclusiones Clave

1.  **Superioridad del Transfer Learning**: El modelo BERT preentrenado superó significativamente a los modelos entrenados desde cero, mejorando especialmente en la detección de emociones minoritarias como *miedo* o *asco*.
2.  **Atención vs Recurrencia**: El modelo Transformer base superó a la GRU, demostrando que los mecanismos de auto-atención capturan mejor el contexto en frases cortas e informales de Reddit.
3.  **Eficiencia en Fine-Tuning**: Se demostró que descongelar solo las últimas 3 capas de BERT ofrece un equilibrio óptimo entre rendimiento (68.45%) y coste computacional, utilizando solo una fracción de los parámetros entrenables.
4.  **Límites del Lenguaje**: Los resultados (~69% acc) reflejan la realidad del NLP en entornos ruidosos: el sarcasmo y la subjetividad humana imponen un techo natural a la clasificación automatizada.

## ⚙️ Instalación y Uso

1. Clonar el repositorio:
   ```bash
   git clone [https://github.com/Javier-MR-Leon/goemotions-nlp-transformers.git](https://github.com/Javier-MR-Leon/goemotions-nlp-transformers.git)
