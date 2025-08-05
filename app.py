import gradio as gr
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# Load model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
labels = ['Negative', 'Neutral', 'Positive']

# Function for single review
def analyze_single_review(text):
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**encoded_input)
    scores = softmax(output.logits.numpy()[0])
    result = {label: round(float(score), 4) for label, score in zip(labels, scores)}
    return result

# Function for CSV upload
def analyze_csv(file):
    df = pd.read_csv(file)

    if 'review' not in df.columns:
        return "❌ Error: CSV must contain a column named 'review'", None

    sentiments = []
    confidences = []

    for review in df['review']:
        encoded_input = tokenizer(str(review), return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            output = model(**encoded_input)
        scores = softmax(output.logits.numpy()[0])
        pred_idx = scores.argmax()
        sentiments.append(labels[pred_idx])
        confidences.append(round(float(scores[pred_idx]), 4))

    df['Sentiment'] = sentiments
    df['Confidence'] = confidences

    output_path = "sentiment_output.csv"
    df.to_csv(output_path, index=False)

    return df, output_path

# Create Gradio tabs
with gr.Blocks(title="Amazon Review Sentiment Analyzer") as demo:
    gr.Markdown("# 📦 Amazon Review Sentiment Analyzer")
    gr.Markdown("Analyze the sentiment of a single review or upload a CSV file with multiple reviews.")

    with gr.Tab("Single Review"):
        single_input = gr.Textbox(lines=3, placeholder="Type your review here", label="Review")
        single_output = gr.Label(label="Sentiment Scores")
        single_btn = gr.Button("Analyze")
        single_btn.click(analyze_single_review, inputs=single_input, outputs=single_output)

    with gr.Tab("CSV Upload"):
        csv_input = gr.File(label="Upload CSV with a 'review' column")
        csv_table = gr.Dataframe(label="Results Table")
        csv_file = gr.File(label="Download Updated CSV")
        csv_btn = gr.Button("Analyze CSV")
        csv_btn.click(analyze_csv, inputs=csv_input, outputs=[csv_table, csv_file])

demo.launch()
