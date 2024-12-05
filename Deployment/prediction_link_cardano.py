import streamlit as st
import joblib
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import requests
import re
import yfinance as yf
from datetime import datetime, timedelta

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and multiple spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('\n', ' ').replace('\t', ' ')
    # Remove extra spaces
    text = ' '.join(text.split())
    return text

def extract_date_from_url(url):
    # Use regex to extract the date from the URL
    match = re.search(r'/(\d{4}/\d{2}/\d{2})/', url)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y/%m/%d').date()
    return None

def get_eth_usd_price_on_date(date):
    # Fetch historical data for ETH-USD
    eth_data = yf.download('ADA-USD', start=date, end=date + timedelta(days=1))
    if not eth_data.empty:
        return eth_data['Close'][0]
    return None



# Initialize Vader Sentiment Analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Initialize FinBert Tokenizer and Model
finbert_tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
finbert_model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

def get_sentiment_score(text):
    text = preprocess_text(text)
    return vader_analyzer.polarity_scores(text)['compound']

def split_into_chunks(text, chunk_size=512):
    tokens = finbert_tokenizer.tokenize(text)
    chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
    return chunks

def get_finbert_sentiment_for_long_text(text):
    text = preprocess_text(text)
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

def extract_coindesk_article_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    header = soup.find('h1', class_='text-headline-lg').get_text(strip=True) if soup.find('h1', class_='text-headline-lg') else ""
    subheader = soup.find('h2', class_='text-body-large text-charcoal-600').get_text(strip=True) if soup.find('h2', class_='text-body-large text-charcoal-600') else ""
    paragraphs = soup.find_all('p')
    
    content = "\n".join(p.get_text(strip=True) for p in paragraphs)
    content = preprocess_text(content)
    
    return header, subheader, content



# Load your model using joblib
model = joblib.load('decision_tree_gradient_boosting_Cardano_V2.pkl')

mean_price_movement = -0.0009074429014943693  
std_price_movement = 0.02102401042451418 




def run():
    theme = ''
    price = 0
    st.header("Prediction Link (Cardano ADA)")
    
    st.header("Article Information")
    article_url = st.text_input('Enter Article URL')
    article_header, article_subheader, article_content = "", "", ""
    
    if article_url:
        article_header, article_subheader, article_content = extract_coindesk_article_content(article_url)
        date = extract_date_from_url(article_url)
        price = get_eth_usd_price_on_date(date)
        theme = re.search(r'www.coindesk.com/(.*?)/', article_url).group(1)
    st.header("Bitcoin Article Information")
    bitcoin_article_url = st.text_input('Enter Bitcoin Article URL')
    bitcoin_article_header, bitcoin_article_subheader, bitcoin_article_content = "", "", ""
    
    if bitcoin_article_url:
        bitcoin_article_header, bitcoin_article_subheader, bitcoin_article_content = extract_coindesk_article_content(bitcoin_article_url)
    else:
# Assigning values to 0
        bitcoin_sentiment_score_header = 0
        bitcoin_sentiment_score_subheader = 0
        bitcoin_sentiment_score_content = 0
        bitcoin_finbert_content_positive = 0
        bitcoin_finbert_content_neutral = 0
        bitcoin_finbert_content_negative = 0
        bitcoin_finbert_header_positive = 0
        bitcoin_finbert_header_neutral = 0
        bitcoin_finbert_header_negative = 0
        bitcoin_finbert_subheader_positive = 0
        bitcoin_finbert_subheader_neutral = 0
        bitcoin_finbert_subheader_negative = 0

    if st.button('Submit'):
        # Calculate sentiment scores using Vader for Ethereum article
        sentiment_score_header = get_sentiment_score(article_header)
        sentiment_score_subheader = get_sentiment_score(article_subheader)
        sentiment_score_content = get_sentiment_score(article_content)

        # Calculate sentiment scores using FinBert for Ethereum article
        finbert_content_positive, finbert_content_neutral, finbert_content_negative = get_finbert_sentiment_for_long_text(article_content)
        finbert_header_positive, finbert_header_neutral, finbert_header_negative = get_finbert_sentiment_for_long_text(article_header)
        finbert_subheader_positive, finbert_subheader_neutral, finbert_subheader_negative = get_finbert_sentiment_for_long_text(article_subheader)

        finbert_sentiment_score_content = finbert_content_positive - finbert_content_negative
        finbert_sentiment_score_header = finbert_header_positive - finbert_header_negative
        finbert_sentiment_score_subheader = finbert_subheader_positive - finbert_subheader_negative

        # Calculate sentiment scores using Vader for Bitcoin article
        bitcoin_sentiment_score_header = get_sentiment_score(bitcoin_article_header)
        bitcoin_sentiment_score_subheader = get_sentiment_score(bitcoin_article_subheader)
        bitcoin_sentiment_score_content = get_sentiment_score(bitcoin_article_content)

        # Calculate sentiment scores using FinBert for Bitcoin article
        bitcoin_finbert_content_positive, bitcoin_finbert_content_neutral, bitcoin_finbert_content_negative = get_finbert_sentiment_for_long_text(bitcoin_article_content)
        bitcoin_finbert_header_positive, bitcoin_finbert_header_neutral, bitcoin_finbert_header_negative = get_finbert_sentiment_for_long_text(bitcoin_article_header)
        bitcoin_finbert_subheader_positive, bitcoin_finbert_subheader_neutral, bitcoin_finbert_subheader_negative = get_finbert_sentiment_for_long_text(bitcoin_article_subheader)

        bitcoin_finbert_sentiment_score_content = bitcoin_finbert_content_positive - bitcoin_finbert_content_negative
        bitcoin_finbert_sentiment_score_header = bitcoin_finbert_header_positive - bitcoin_finbert_header_negative
        bitcoin_finbert_sentiment_score_subheader = bitcoin_finbert_subheader_positive - bitcoin_finbert_subheader_negative

        st.header("Model Inputs")
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
            'Theme_markets': 1 if theme == 'markets' else 0,
            'Theme_tech': 1 if theme == 'business' else 0
        }
        
        user_input = pd.DataFrame(data, index=[0])

        # Make predictions
        if not article_content.strip():
            st.warning("Please provide valid content for both articles to make a prediction.")
        else:
            z_score_prediction = model.predict(user_input)[0]

            # Denormalize the prediction
            actual_price_movement = (z_score_prediction * std_price_movement) + mean_price_movement

            # Display z-score and denormalized prediction
            st.write('Predicted Price Movement (USD):')
            st.write(actual_price_movement)
            st.write("Current Cardano ADA Price (USD)")
            st.write(price)
            st.write("Price Prediction (USD)")
            st.write(price + actual_price_movement)