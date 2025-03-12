from transformers import pipeline
import gradio as gr

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Define a function for sentiment analysis
def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return f"Label: {result['label']}, Confidence: {result['score']:.4f}"

# Create Gradio Interface
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(placeholder="Enter text here..."),
    outputs="text",
    title="Sentiment Analysis",
    description="Enter a sentence, and the model will predict whether it is positive or negative."
)

# Launch the app
iface.launch()
