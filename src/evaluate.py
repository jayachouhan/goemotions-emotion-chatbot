
# # print("Evaluation file ready")
# # Model evaluation file

# import pandas as pd
# import torch
# import pickle

# from sklearn.metrics import (
#     accuracy_score,
#     f1_score,
#     confusion_matrix,
#     classification_report
# )

# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification
# )

# # Load model
# model = AutoModelForSequenceClassification.from_pretrained(
#     "models/saved_model"
# )

# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(
#     "ai4bharat/IndicBERTv2-MLM-only"
# )

# # Load label encoder
# with open("models/label_encoder.pkl", "rb") as f:
#     encoder = pickle.load(f)

# # Load test data
# df = pd.read_csv("data/processed/test.csv")

# texts = df["text"].tolist()
# true_labels = df["label"].tolist()

# predictions = []

# # Predict all test samples
# for text in texts:

#     inputs = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=128
#     )

#     with torch.no_grad():
#         outputs = model(**inputs)

#     pred = torch.argmax(outputs.logits, dim=1).item()

#     predictions.append(pred)

# # Calculate metrics
# accuracy = accuracy_score(true_labels, predictions)

# f1 = f1_score(
#     true_labels,
#     predictions,
#     average="weighted"
# )

# # Print results
# print(f"Accuracy: {accuracy:.2f}")

# print(f"F1-Score: {f1:.2f}")

# print("\nClassification Report:\n")

# print(
#     classification_report(
#         true_labels,
#         predictions
#     )
# )

# # Confusion matrix
# cm = confusion_matrix(
#     true_labels,
#     predictions
# )

# print("\nConfusion Matrix:\n")

# print(cm)
import os
import pandas as pd
import torch
import joblib
import json

from sklearn.metrics import (
    accuracy_score,
    f1_score
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "models/saved_model"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "ai4bharat/IndicBERTv2-MLM-only"
)

# Load encoder
encoder = joblib.load(
    "models/label_encoder.pkl"
)

# Load test data
df = pd.read_csv("data/processed/test.csv")

texts = df["text"].tolist()

true_labels = df["label_encoded"].tolist()

predictions = []

# Predict all test samples
for text in texts:

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=1).item()

    predictions.append(pred)

# Calculate metrics
accuracy = accuracy_score(
    true_labels,
    predictions
)

f1 = f1_score(
    true_labels,
    predictions,
    average="weighted"
)

# Save metrics
metrics = {
    "accuracy": round(accuracy * 100, 2),
    "f1_score": round(f1 * 100, 2)
}
os.makedirs("outputs", exist_ok=True)

with open("outputs/metrics.json", "w") as f:
    json.dump(metrics, f)

print("Metrics saved successfully")
