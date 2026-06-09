# # ============================================
# # Streamlit Frontend
# # ============================================

# import os
# import sys
# import json

# import streamlit as st
# import pandas as pd

# # Add src folder path

# sys.path.append(
#     os.path.abspath(
#         os.path.join(
#             os.path.dirname(__file__),
#             '..',
#             'src'
#         )
#     )
# )

# # Import custom files

# from inference import predict_emotion

# from graphs import create_emotion_graph

# from llm_analysis import analyze_therapist_response


# # ============================================
# # Page Config
# # ============================================

# st.set_page_config(
#     page_title="Emotion-Aware Therapy Monitoring",
#     layout="wide"
# )

# # ============================================
# # Title
# # ============================================

# st.title(
#     "🧠 Turn-by-Turn Emotion-Aware Therapy Monitoring"
# )

# # ============================================
# # Conversation Input
# # ============================================

# conversation = st.text_area(
#     "Enter conversation (one line = one turn)",
#     height=300
# )

# # ============================================
# # Therapist Response Input
# # ============================================

# therapist_response = st.text_area(
#     "Enter therapist response",
#     height=150
# )

# # ============================================
# # Analyze Button
# # ============================================

# if st.button("Analyze"):

#     # Split conversation into turns

#     turns = conversation.split("\n")

#     # Store results

#     results = []

#     # Process each turn

#     for idx, turn in enumerate(turns):

#         # Skip empty lines

#         if turn.strip() == "":
#             continue

#         # Predict emotion

#         emotion = predict_emotion(turn)

#         # Store prediction

#         results.append({

#             "Turn": idx + 1,

#             "Text": turn,

#             "Emotion": emotion
#         })

#     # ============================================
#     # Create DataFrame
#     # ============================================

#     df = pd.DataFrame(results)

#     # ============================================
#     # Show Predictions
#     # ============================================

#     st.subheader(
#         "Turn-by-Turn Emotion Prediction"
#     )

#     st.dataframe(df)

#     # ============================================
#     # Save Predictions CSV
#     # ============================================

#     os.makedirs(
#         "outputs",
#         exist_ok=True
#     )

#     df.to_csv(
#         "outputs/predictions.csv",
#         index=False
#     )

#     # ============================================
#     # Emotion Graph
#     # ============================================

#     st.subheader(
#         "Emotion Flow Graph"
#     )

#     fig = create_emotion_graph(df)

#     st.plotly_chart(
#         fig,
#         use_container_width=True
#     )

#     # ============================================
#     # Overall Emotion
#     # ============================================

#     overall_emotion = df["Emotion"].mode()[0]

#     st.subheader(
#         "Overall Emotional State"
#     )

#     st.success(
#         f"Overall Dominant Emotion: {overall_emotion}"
#     )

#     # ============================================
#     # Therapist Analysis
#     # ============================================

#     st.subheader(
#         "Therapist Response Quality Analysis"
#     )

#     analysis = analyze_therapist_response(
#         conversation,
#         therapist_response
#     )

#     st.write(analysis)

# # ============================================
# # Model Metrics
# # ============================================

# try:

#     with open(
#         "outputs/metrics.json",
#         "r"
#     ) as f:

#         metrics = json.load(f)

#     st.subheader(
#         "Model Performance"
#     )

#     st.success(
#         f"Accuracy: {metrics['accuracy']}%"
#     )

#     st.info(
#         f"F1-Score: {metrics['f1_score']}%"
#     )

#     st.write(
#         "Evaluation Tool Used: scikit-learn"
#     )

# except:

#     st.warning(
#         "Run evaluate.py first"
#     )


# ============================================
# Streamlit Frontend
# File: app/streamlit_app.py
# ============================================

import os
import sys
import json

import streamlit as st
import pandas as pd

# ============================================
# Add src folder path
# ============================================

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            'src'
        )
    )
)

# ============================================
# Import custom files
# ============================================

from inference import predict_emotion

from graphs import create_emotion_graph

from llm_analysis import analyze_therapist_response


# ============================================
# Streamlit Page Config
# ============================================

st.set_page_config(
    page_title="Emotion-Aware Therapy Monitoring",
    layout="wide"
)

# ============================================
# Title
# ============================================

st.title(
    "🧠 Turn-by-Turn Emotion-Aware Therapy Monitoring"
)

# ============================================
# Conversation Input
# ============================================

conversation = st.text_area(
    "Enter conversation (one line = one turn)",
    height=300
)

# ============================================
# Therapist Response Input
# ============================================

therapist_response = st.text_area(
    "Enter therapist response",
    height=150
)

# ============================================
# Analyze Button
# ============================================

if st.button("Analyze"):

    # Split conversation into turns

    turns = conversation.split("\n")

    # Store prediction results

    results = []

    # ============================================
    # Predict Emotion for Each Turn
    # ============================================

    for idx, turn in enumerate(turns):

        # Skip empty lines

        if turn.strip() == "":
            continue

        # Predict emotion

        emotion = predict_emotion(turn)

        # Store prediction

        results.append({

            "Turn": idx + 1,

            "Text": turn,

            "Emotion": emotion
        })

    # ============================================
    # Create DataFrame
    # ============================================

    df = pd.DataFrame(results)

    # ============================================
    # Calculate Overall Emotion
    # ============================================

    if len(df) > 0:

        overall_emotion = df["Emotion"].mode()[0]

        # Create overall row

        overall_row = pd.DataFrame([{

            "Turn": "Overall",

            "Text": "Final Emotional State",

            "Emotion": overall_emotion
        }])

        # Add overall row

        df = pd.concat(
            [df, overall_row],
            ignore_index=True
        )

    # ============================================
    # Show Predictions Table
    # ============================================

    st.subheader(
        "Turn-by-Turn Emotion Prediction"
    )

    st.dataframe(df)

    # ============================================
    # Save CSV File
    # ============================================

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    df.to_csv(
        "outputs/predictions.csv",
        index=False
    )

    # ============================================
    # Graph Visualization
    # ============================================

    # Remove overall row for graph

    graph_df = df[
        df["Turn"] != "Overall"
    ]

    st.subheader(
        "Emotion Flow Graph"
    )

    fig = create_emotion_graph(
        graph_df
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ============================================
    # Overall Emotion Display
    # ============================================

    if len(df) > 0:

        st.subheader(
            "Overall Emotional State"
        )

        st.success(
            f"Overall Dominant Emotion: {overall_emotion}"
        )

    # ============================================
    # Therapist Response Analysis
    # ============================================

    st.subheader(
        "Therapist Response Quality Analysis"
    )

    analysis = analyze_therapist_response(
        conversation,
        therapist_response
    )

    st.write(analysis)

# ============================================
# Model Metrics Section
# ============================================

try:

    with open(
        "outputs/metrics.json",
        "r"
    ) as f:

        metrics = json.load(f)

    st.subheader(
        "Model Performance"
    )

    st.success(
        f"Accuracy: {metrics['accuracy']}%"
    )

    st.info(
        f"F1-Score: {metrics['f1_score']}%"
    )

    st.write(
        "Evaluation Tool Used: scikit-learn"
    )

except:

    st.warning(
        "Run evaluate.py first"
    )