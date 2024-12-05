import streamlit as st
import joblib
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import nltk
from nltk.corpus import stopwords

# Download the stopwords for NLTK if you haven't already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize Vader Sentiment Analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Initialize FinBert Tokenizer and Model
finbert_tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
finbert_model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

def get_sentiment_score(text):
    return vader_analyzer.polarity_scores(text)['compound']

def split_into_chunks(text, chunk_size=512):
    tokens = finbert_tokenizer.tokenize(text)
    chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
    return chunks

def get_finbert_sentiment_for_long_text(text):
    chunks = split_into_chunks(text)
    if len(chunks) == 0:
        return 0, 0, 0  # Return neutral scores if there are no chunks
    
    sentiments = []
    for chunk in chunks:
        chunk_text = finbert_tokenizer.convert_tokens_to_string(chunk)
        sentiments.append(get_finbert_sentiment(chunk_text))

    avg_positive = sum([s[0] for s in sentiments]) / len(sentiments)
    avg_neutral = sum([s[1] for s in sentiments]) / len(sentiments)
    avg_negative = sum([s[2] for s in sentiments]) / len(sentiments)

    return avg_positive, avg_neutral, avg_negative

def get_finbert_sentiment(text):
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = finbert_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive_score = probs[0][1].item()
    neutral_score = probs[0][0].item()
    negative_score = probs[0][2].item()
    return positive_score, neutral_score, negative_score

# Load your model using joblib
model = joblib.load('Deployment/decision_tree_gradient_boosting_Ethereum_V2.pkl')

# Placeholder for mean and standard deviation of the original price movements
mean_price_movement = 0.0  # Replace with actual mean value
std_price_movement = 1.0   # Replace with actual standard deviation value

def run():
    st.header("Prediction (Ethereum)")
    
    st.header("Article Information")
    article_header = st.text_input('Article Header')
    article_subheader = st.text_input('Article SubHeader')
    article_body = st.text_area('Article Body')
    
    st.header("Bitcoin Article Information")
    bitcoin_article_header = st.text_input('Bitcoin Article Header')
    bitcoin_article_subheader = st.text_input('Bitcoin Article SubHeader')
    bitcoin_article_body = st.text_area('Bitcoin Article Body')
    
    # Calculate sentiment scores using Vader for Ethereum article
    sentiment_score_header = get_sentiment_score(article_header)
    sentiment_score_subheader = get_sentiment_score(article_subheader)
    sentiment_score_content = get_sentiment_score(article_body)

    # Calculate sentiment scores using FinBert for Ethereum article
    finbert_content_positive, finbert_content_neutral, finbert_content_negative = get_finbert_sentiment_for_long_text(article_body)
    finbert_header_positive, finbert_header_neutral, finbert_header_negative = get_finbert_sentiment_for_long_text(article_header)
    finbert_subheader_positive, finbert_subheader_neutral, finbert_subheader_negative = get_finbert_sentiment_for_long_text(article_subheader)

    finbert_sentiment_score_content = finbert_content_positive - finbert_content_negative
    finbert_sentiment_score_header = finbert_header_positive - finbert_header_negative
    finbert_sentiment_score_subheader = finbert_subheader_positive - finbert_subheader_negative

    # Calculate sentiment scores using Vader for Bitcoin article
    bitcoin_sentiment_score_header = get_sentiment_score(bitcoin_article_header)
    bitcoin_sentiment_score_subheader = get_sentiment_score(bitcoin_article_subheader)
    bitcoin_sentiment_score_content = get_sentiment_score(bitcoin_article_body)

    # Calculate sentiment scores using FinBert for Bitcoin article
    bitcoin_finbert_content_positive, bitcoin_finbert_content_neutral, bitcoin_finbert_content_negative = get_finbert_sentiment_for_long_text(bitcoin_article_body)
    bitcoin_finbert_header_positive, bitcoin_finbert_header_neutral, bitcoin_finbert_header_negative = get_finbert_sentiment_for_long_text(bitcoin_article_header)
    bitcoin_finbert_subheader_positive, bitcoin_finbert_subheader_neutral, bitcoin_finbert_subheader_negative = get_finbert_sentiment_for_long_text(bitcoin_article_subheader)

    bitcoin_finbert_sentiment_score_content = bitcoin_finbert_content_positive - bitcoin_finbert_content_negative
    bitcoin_finbert_sentiment_score_header = bitcoin_finbert_header_positive - bitcoin_finbert_header_negative
    bitcoin_finbert_sentiment_score_subheader = bitcoin_finbert_subheader_positive - bitcoin_finbert_subheader_negative

    st.header("Model Inputs")
    theme = st.radio(
        "Select Theme",
        ('Theme_business', 'Theme_markets', 'Theme_policy', 'Theme_tech')
    )
    data = {
        'Finbert_Sentiment_Score_Content': finbert_sentiment_score_content,
        'Finbert_Sentiment_Score_Header': finbert_sentiment_score_header,
        'Finbert_Sentiment_Score_Subheader': finbert_sentiment_score_subheader,
        'Sentiment_Score_Content': sentiment_score_content,
        'Sentiment_Score_Header': sentiment_score_header,
        'Sentiment_Score_SubHeader': sentiment_score_subheader,
        'Bitcoin_Finbert_Sentiment_Score_Content': bitcoin_finbert_sentiment_score_content,
        'Bitcoin_Finbert_Sentiment_Score_Header': bitcoin_finbert_sentiment_score_header,
        'Bitcoin_Finbert_Sentiment_Score_Subheader': bitcoin_finbert_sentiment_score_subheader,
        'Bitcoin_Sentiment_Score_Content': bitcoin_sentiment_score_content,
        'Bitcoin_Sentiment_Score_Header': bitcoin_sentiment_score_header,
        'Bitcoin_Sentiment_Score_SubHeader': bitcoin_sentiment_score_subheader,
        'Theme_business': 1 if theme == 'Theme_business' else 0,
        'Theme_markets': 1 if theme == 'Theme_markets' else 0,
        'Theme_policy': 1 if theme == 'Theme_policy' else 0,
        'Theme_tech': 1 if theme == 'Theme_tech' else 0,
    }
    
    user_input = pd.DataFrame(data, index=[0])

    # Display user input
    st.write('User Input:')
    st.write(user_input)

    # Make predictions
    z_score_prediction = model.predict(user_input)[0]

    # Denormalize the prediction
    actual_price_movement = (z_score_prediction * std_price_movement) + mean_price_movement

    # Display z-score and denormalized prediction
    st.write('Z-Score Prediction:')
    st.write(z_score_prediction)
    st.write('Actual Price Movement Prediction:')
    st.write(actual_price_movement)
