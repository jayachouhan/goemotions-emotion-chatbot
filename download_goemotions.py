from datasets import load_dataset
import pandas as pd

dataset = load_dataset("go_emotions")

rows = []

for item in dataset["train"]:
    if len(item["labels"]) > 0:
        rows.append({
            "text": item["text"],
            "label": item["labels"][0]
        })

df = pd.DataFrame(rows)

df.to_csv(
    "data/raw/goemotions_dataset.csv",
    index=False
)

print("Saved:", df.shape)