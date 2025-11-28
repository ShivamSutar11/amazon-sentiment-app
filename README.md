📘 Amazon Review Sentiment Analysis — Fine-Tuned DistilBERT + Streamlit App

This project fine-tunes DistilBERT, a lightweight transformer model, on the Amazon Polarity dataset to classify product reviews into Positive or Negative sentiment.
The trained model is deployed locally using Streamlit, giving you an interactive UI where users can type reviews and instantly see predictions with confidence scores.

🚀 Features

✅ Fine-tuned DistilBERT for sentiment classification

✅ Uses HuggingFace Transformers + Datasets

✅ End-to-end workflow: dataset → training → evaluation → saving model

✅ Fully interactive Streamlit web app

✅ Offline local deployment

✅ Clean and simple UI

✅ Confidence score returned with every prediction

📂 Project Structure
amazon_sentiment/
│── app.py                     # Streamlit app
│── amazon_sentiment_finetuned/
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer.json
│       ├── vocab.txt
│       └── ... model files
│── requirements.txt
└── README.md

🧠 Model Details

Base Model: distilbert-base-uncased

Task: Binary Sentiment Classification

Dataset: Amazon Polarity (HuggingFace Datasets)

Metrics: Accuracy (can add F1 if needed)

Training:

Batch size: 16

Epochs: 1 (for fast demo)

Optimized on GPU (NVIDIA T4)

🧪 How to Run Locally
1. Clone the repo
git clone https://github.com/your-username/amazon-sentiment-app.git
cd amazon-sentiment-app

2. Install dependencies
pip install -r requirements.txt

3. Run the app
streamlit run app.py


The app will open in your browser at:

http://localhost:8501

🖥️ Streamlit UI Preview

The app includes:

A text area to type or paste an Amazon review

A button to trigger the prediction

Output showing

Sentiment: Positive/Negative

Confidence score