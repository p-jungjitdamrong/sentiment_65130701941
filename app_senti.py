

import streamlit as st
from transformers import pipeline

# Load the sentiment analysis model
model_name = "poom-sci/WangchanBERTa-finetuned-sentiment"

# Load the saved pipeline model
tf_model = joblib.load('sentiment_pipeline_TF_IDF.pkl')
sentiment_tf = pipeline('sentiment-analysis', model=tf_model)

sentiment_huggin = pipeline('sentiment-analysis', model=model_name)


# Streamlit app
st.title("Thai Sentiment Analysis App")

# Input text
text_input = st.text_area("Enter Thai text for sentiment analysis", "ขอความเห็นหน่อย... ")

# Button to trigger analysis
if st.button("Analyze Sentiment"):
    # Analyze sentiment using the model
    results = sentiment_analyzer_loaded([text_input])

    # Extract sentiment and score
    sentiment = results[0]['label']
    score = results[0]['score']
    
    # Display result as progress bars
    st.subheader("Sentiment Analysis Result with Hugging Face:")

    if sentiment == 'pos':
        st.success(f"Positive Sentiment (Score: {score:.2f})")
        st.progress(score)
    elif sentiment == 'neg':
        st.error(f"Negative Sentiment (Score: {score:.2f})")
        st.progress(score)
    else:
        st.warning(f"Neutral Sentiment (Score: {score:.2f})")
        st.progress(score)

    # Analyze sentiment using the model
    results = sentiment_pipeline_loaded([text_input])

    # Extract sentiment and score
    sentiment = results[0]['label']
    score = results[0]['score']
    
    # Display result as progress bars
    st.subheader("Sentiment Analysis Result with TF-IDF:")

    if sentiment == 'positive':
        st.success(f"Positive Sentiment")
        st.progress(score)
    elif sentiment == 'negative':
        st.error(f"Negative Sentiment")
        st.progress(score)
    else:
        st.warning(f"Neutral Sentiment")
        st.progress(score)
