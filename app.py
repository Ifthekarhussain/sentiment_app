import gradio as gr
import joblib
import pandas as pd

# Load the best model
model = joblib.load("best_LogisticRegression.joblib")

# Load BoW feature columns
bow_features = pd.read_csv("Homework01_BoW.csv").columns

# Prediction function
def predict_sentiment(user_input):
    input_df = pd.DataFrame(0, index=[0], columns=bow_features)
    for word in user_input.split():
        if word in input_df.columns:
            input_df[word] = 1
    prediction = model.predict(input_df)[0]
    return "Positive" if prediction == 1 else "Negative"

# Launch the app
app = gr.Interface(fn=predict_sentiment, inputs="text", outputs="text", title="Sentiment Prediction App")
app.launch()

