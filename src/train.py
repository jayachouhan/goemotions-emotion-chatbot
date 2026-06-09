
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import accuracy_score, f1_score

from dataset import EmotionDataset
from config import *

train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/val.csv")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = EmotionDataset(
    train_df["text"].tolist(),
    train_df["label_encoded"].tolist(),
    tokenizer,
    MAX_LENGTH
)

val_dataset = EmotionDataset(
    val_df["text"].tolist(),
    val_df["label_encoded"].tolist(),
    tokenizer,
    MAX_LENGTH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=train_df["label_encoded"].nunique()
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_score": f1_score(labels, predictions, average="weighted")
    }

training_args = TrainingArguments(
    output_dir="./models/saved_model",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model("./models/saved_model")
tokenizer.save_pretrained("./models/tokenizer")

print("Training completed")
