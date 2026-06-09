import torch
import torch.nn.functional as F
import joblib

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

# Load label encoder

encoder = joblib.load(
    "models/label_encoder.pkl"
)

# Prediction function

def predict_emotion(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

    logits = outputs.logits

    probabilities = F.softmax(
        logits,
        dim=1
    )

    pred_id = torch.argmax(
        probabilities,
        dim=1
    ).item()

    predicted_emotion = encoder.inverse_transform(
        [pred_id]
    )[0]

    confidence_scores = {}

    probs = probabilities[0].tolist()

    for i, prob in enumerate(probs):

        emotion = encoder.inverse_transform([i])[0]

        confidence_scores[emotion] = round(
            prob * 100,
            2
        )

    # return {
    #     "emotion": predicted_emotion,
    #     "confidence": confidence_scores
    # }
    return predicted_emotion