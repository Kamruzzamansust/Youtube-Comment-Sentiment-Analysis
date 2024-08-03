import streamlit as st
import pandas as pd
from langchain_ai21 import ChatAI21
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the AI21 API
#os.environ['AI21_API_KEY'] = os.getenv("AI21_API_KEY")
llm = ChatAI21(model = 'jamba-instruct',
               temperature=0,
               max_tokens=60,
               api_key= "Your Api Key"
               ) 

# Define a function for sentiment analysis
def analyze_sentiment(text):
    response = llm.invoke(
        input=f" imagine you are a great sentiment analyst !!. Just assign the sentiment as Positive, Neutral, or Negative for the following text; if you don't Know then Just assign don't know : {text}",
    )
    return response.content

# Streamlit app
st.title("YouTube Comment Sentiment Analysis With AI21 Labs")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    df = df.drop(['Name', 'Time','Likes','Reply Count','Reply Author','Reply','Published','Updated'], axis=1)
    st.header("Original Comment Scrape from Yotube Using Google Sheets App Srcipt")
    st.write(df)

    if 'Comment' in df.columns:
        # Perform sentiment analysis
        df['sentiment'] = df['Comment'].apply(analyze_sentiment)

        # Display the results
        st.write("Sentiment Analysis Results:")
        st.dataframe(df)
    else:
        st.error("The CSV file must contain a 'Comment' column.")
else:
    st.info("Please upload a CSV file to analyze.")



