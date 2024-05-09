import streamlit as st
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import os


# Get the full paths to the pickle files
vectorizer_path = os.path.join(os.path.dirname(__file__), 'rfc_vectorizer.pkl')
model_path = os.path.join(os.path.dirname(__file__), 'rfc_model.pkl')

 #Load both the vectorizer and the model
with open(vectorizer_path, 'rb') as f1, open(model_path, 'rb') as f2:
    tfidf, model = pickle.load(f1), pickle.load(f2)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def text_process(text):
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    
    # Convert tokens to lowercase
    tokens = [word.lower() for word in tokens]
    
    # Convert list of tokens to a single string
    text = ' '.join(tokens)
    
    # Function to remove HTML tags
    def remove_html_tags(text):
        clean_text = re.sub(r'<.*?>', '', text)
        return clean_text
    
    # Function to remove stopwords
    def remove_stopwords(text):
        words = [word for word in text.split() if word.lower() not in stop_words]
        return " ".join(words)
    
    # Function to clean URLs
    def clean_url(text):
        text = re.sub(r"((https:|http|ftp)?(:\/\/)?(www\.)?)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/\/=]*)", ' ', text)
        return re.sub(r'/', ' / ', text)
    
    # Function to clean punctuations
    def clean_punctuations(text):
        return text.translate(str.maketrans('', '', string.punctuation))
    
    # Function to remove repeating characters
    def clean_repeating_char(text):
        return re.sub(r"(.)\1\1+", r"\1\1", text)
    
    # Function to clean numbers
    def clean_numbers(text):
        return re.sub('[0-9]+', '', text)
    
    # Function to remove hashtags
    def remove_hashtag(text):
        return re.sub('#[\w\d]+', ' ', text)
    
    # Function to clean usernames
    def clean_username(text):
        return re.sub('@[^\s]+', ' ', text)
    
    # Function to clean emojis and non-ASCII characters
    def clean_non_ascii(text):
        text = text.encode("ascii", "ignore").decode()
        return text
    
    # Function to remove images
    def remove_images(tweet):
        cleaned_tweet = re.sub(r"pic\.twitter\.com/\S+", '', tweet)
        cleaned_tweet = re.sub("\w+(\.png|\.jpg|\.gif|\.jpeg)", " ", cleaned_tweet)
        return cleaned_tweet
    
    # Function to lemmatize words
    def lemmatize_words(text):
        return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    
    # Apply all preprocessing steps
    text = remove_html_tags(text)
    text = remove_stopwords(text)
    text = clean_url(text)
    text = clean_punctuations(text)
    text = clean_repeating_char(text)
    text = clean_numbers(text)
    text = remove_hashtag(text)
    text = clean_username(text)
    text = clean_non_ascii(text)  
    text = remove_images(text)
    text = lemmatize_words(text)
    
    return text

st.title("Tweet Sentiment Analysis")
input_tweet=st.text_input("Enter your tweet")

if st.button("Predict Sentiment"):
    if input_tweet:
        transformed_tweet=text_process(input_tweet)
        vectorized_tweet=tfidf.transform([transformed_tweet])
        result=model.predict(vectorized_tweet)

        if result==0:
            st.header('The tweet is irrelevant')
        elif result==1:
            st.header('The tweet is Negative')
        elif result==2:
            st.header('The tweet is Neutral')
        else:
            st.header('The tweet is Positive')