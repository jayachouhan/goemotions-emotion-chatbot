
# Emotion-Aware Therapy Monitoring

End-to-end LLM + NLP project for Hindi psychotherapy conversation analysis.

## Features
- IndicBERT-v2 emotion classification
- Turn-by-turn monitoring
- Emotion graphs
- Therapist response quality analysis
- CSV export
- Confusion matrix
- Streamlit dashboard

## Run

```bash
pip install -r requirements.txt
python src/preprocess.py
python src/train.py
python src/evaluate.py
streamlit run app/streamlit_app.py
```
