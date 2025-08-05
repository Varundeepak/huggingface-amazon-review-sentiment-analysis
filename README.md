# 📦 Amazon Review Sentiment Analyzer

A sentiment analysis web app that allows users to analyze Amazon-style product reviews using a RoBERTa-based model from Hugging Face. Built using Gradio for the UI, this app supports both **single review input** and **CSV file upload** for batch analysis

---

## ✨ Features

- 🔍 **Single Review Analysis**  
  Enter a single product review and get the predicted sentiment (Positive, Neutral, or Negative) along with confidence scores

- 📤 **CSV Upload for Bulk Analysis**  
  Upload a `.csv` file with a column named `review`. The app will analyze each row and return a new file with predicted sentiment and confidence scores

- 📥 **Downloadable Results**  
  The processed file with sentiment predictions can be downloaded for reporting or further analysis

---

## 🚀 Live Demo

👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/Varundeepak/amazon-review-sentiment-analyzer)

---

## 🧠 Model Details

- **Model**: [`cardiffnlp/twitter-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)  
- **Architecture**: RoBERTa  
- **Labels**: Positive, Neutral, Negative  
- **Trained on**: Twitter sentiment dataset  
- **Inference**: Done using `transformers`, `torch`, and `scipy`

---

## ⚙️ Tech Stack

- [🤗 Hugging Face Transformers](https://huggingface.co/transformers/)
- [Gradio](https://www.gradio.app/)
- PyTorch
- Pandas
- Scipy

---

## 📌 Usage Notes

- The CSV input must contain a column named exactly review
- Long reviews are automatically truncated at 512 tokens to fit the model’s limit

---

## 📚 Learnings & Takeaways

- How to use pretrained models from Hugging Face for sentiment analysis
- Working with tokenizers, logits, and softmax for inference
- Building Gradio UIs with multiple input modes
- Deploying to Hugging Face Spaces
