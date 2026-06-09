# ============================================
# Preprocessing Script
# File: src/preprocess.py
# ============================================

import os
import pandas as pd

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# import joblib


# ============================================
# Text Cleaning Function
# ============================================

def clean_text(text):

    """
    Clean input text
    """

    return str(text).strip()


# ============================================
# Create Required Folders
# ============================================

os.makedirs("models", exist_ok=True)

os.makedirs("data/processed", exist_ok=True)


# ============================================
# Load Dataset
# ============================================

df = pd.read_csv(
    "data/raw/goemotions_dataset.csv"
)


# ============================================
# Clean Text Column
# ============================================

df["text"] = df["text"].apply(
    clean_text
)


# ============================================
# Encode Emotion Labels
# ============================================

# encoder = LabelEncoder()

# df["label_encoded"] = df["label"]
df["label_encoded"] = df["label"].astype(int)


# ============================================
# Save Label Encoder
# ============================================

# joblib.dump(
#     encoder,
#     "models/label_encoder.pkl"
# )


# ============================================
# Train / Validation / Test Split
# ============================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label_encoded"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["label_encoded"]
)


# ============================================
# Save Processed CSV Files
# ============================================

train_df.to_csv(
    "data/processed/train.csv",
    index=False
)

val_df.to_csv(
    "data/processed/val.csv",
    index=False
)

test_df.to_csv(
    "data/processed/test.csv",
    index=False
)


# ============================================
# Print Dataset Information
# ============================================

print("\nPreprocessing completed successfully\n")

print("Train Shape :", train_df.shape)

print("Validation Shape :", val_df.shape)

print("Test Shape :", test_df.shape)

print("\nLabel Classes:\n")

# print(encoder.classes_)
print("\nGoEmotions preprocessing done ✔")
print("Unique labels:", df["label"].nunique())