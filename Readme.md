# LLM Fine-Tuning for Toxicity Classification

## Project Overview

This project demonstrates fine-tuning a pre-trained transformer-based Large Language Model (LLM) for multi-class text classification using the Hugging Face ecosystem.

The model is trained to classify text into toxicity-related categories using a publicly available dataset. The implementation covers dataset preparation, preprocessing, tokenization, model fine-tuning, evaluation, and inference pipeline creation.

This project reflects practical experience with transfer learning, transformer architectures, and production-ready NLP workflows.

---

## Objective

To fine-tune a pre-trained BERT model for multi-class toxic speech detection and evaluate its performance using standard classification metrics.

---

## Tech Stack

- Python 3.x  
- Hugging Face Transformers  
- Hugging Face Datasets  
- PyTorch  
- Scikit-learn  
- NumPy  
- Pandas  

---

## Dataset

The project uses the `hate_speech_offensive` dataset from Hugging Face.

The dataset contains text samples categorized into:

- Hate Speech  
- Offensive Language  
- Neither  

The dataset is:
- Split into training, validation, and test sets  
- Converted into Pandas DataFrames for preprocessing  
- Reformatted into Hugging Face Dataset format for training  

---

## Project Workflow

### 1. Dataset Loading

```python
from datasets import load_dataset
dataset = load_dataset("hate_speech_offensive")
```

### 2. Label Mapping

- String labels are mapped to numerical IDs.
- `label2id` and `id2label` dictionaries are created for model compatibility.

### 3. Tokenization

Model checkpoint used:

```
bert-base-uncased
```

Tokenizer initialization:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

### 4. Model Initialization

```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
```

### 5. Training Configuration

Training is handled using Hugging Face’s `Trainer` API with:

- Epochs: 3  
- Batch size: 16  
- Weight decay: 0.01  
- Warmup steps: 500  
- Evaluation during training  

### 6. Evaluation Metrics

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
```

Metrics used:
- Accuracy  
- Precision (weighted)  
- Recall (weighted)  
- F1 Score (weighted)  

### 7. Model Training

```python
trainer.train()
```

Model and tokenizer are saved after training.

### 8. Model Evaluation

```python
trainer.evaluate()
```

### 9. Inference Pipeline

```python
from transformers import pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
classifier("Example input text")
```

---

## Project Structure

```
LLM-Fine-Tuning/
│
├── LLM Fine Tuning.ipynb
├── fine_tuned_model/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer files
│
├── results/
└── README.md
```

---

## Installation

```bash
pip install transformers datasets torch scikit-learn pandas numpy
```

---

## How to Run

1. Clone the repository  
2. Install dependencies  
3. Open `LLM Fine Tuning.ipynb`  
4. Run all cells sequentially  

---

## Key Learning Outcomes

- Fine-tuning transformer-based LLMs  
- Transfer learning for NLP  
- Hugging Face Trainer API usage  
- Evaluation pipeline implementation  
- Model saving and reloading  
- Building inference-ready pipelines  

---

## Future Improvements

- Hyperparameter tuning  
- Early stopping  
- Cross-validation  
- Class imbalance handling  
- Confusion matrix visualization  
- Deployment with FastAPI or Streamlit  
- Experiment tracking with MLflow or Weights & Biases  

---

## Author

Anirudh  
  
