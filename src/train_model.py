import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# =========================
# Load dataset
# =========================
train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/val.csv")

# =========================
# Convert to HF Dataset
# =========================
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# =========================
# Model + Tokenizer
# =========================
model_name = "bert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=28
)

# =========================
# Tokenization
# =========================
def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label_encoded"])

train_dataset = train_dataset.rename_column("label_encoded", "labels")
val_dataset = val_dataset.rename_column("label_encoded", "labels")

val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# =========================
# Metrics
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# =========================
# Training args (LIGHT for laptop)
# =========================
training_args = TrainingArguments(
    output_dir="./models/indicbert",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,   # IMPORTANT: only 1 epoch (fast)
    weight_decay=0.01,
    logging_dir="./logs"
)

# =========================
# Trainer
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=28,
    ignore_mismatched_sizes=True
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# =========================
# Train
# =========================
trainer.train()

# =========================
# Save model
# =========================
model.save_pretrained("./models/indicbert_emotion")
tokenizer.save_pretrained("./models/indicbert_emotion")

print("Training Completed ✔")