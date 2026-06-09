import plotly.express as px

def create_emotion_graph(df):

    fig = px.line(
        df,
        x="Turn",
        y="Emotion",
        markers=True,
        title="Turn-by-Turn Emotion Flow"
    )

    return fig