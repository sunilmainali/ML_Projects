import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
import warnings


warnings.filterwarnings("ignore", category=UserWarning)




ps = PorterStemmer()

def transform_text(Text):
    Text = Text.lower()
    Text = nltk.word_tokenize(Text)
    
    y = []
    for i in Text:
        if i.isalnum():
            y.append(i)
    Text = y[:]
    y.clear()
    
    for i in Text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    Text = y[:]
    y.clear()
    
    for i in Text:
        y.append(ps.stem(i))
        
    return " ".join(y)





# Get the full paths to the pickle files
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

 #Load both the vectorizer and the model
with open(vectorizer_path, 'rb') as f1, open(model_path, 'rb') as f2:
    tfidf, model = pickle.load(f1), pickle.load(f2)

#tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
#model = pickle.load(open('model.pkl', 'rb'))

st.title("Email Spam Classifier")
input_email = st.text_input("Enter the email: ")

if st.button("predict"):
    if input_email:
        transform_email = transform_text(input_email)
        vector_input = tfidf.transform([transform_email])

        # Convert sparse matrix to dense matrix
        vector_input_dense = vector_input.toarray()

        result = model.predict(vector_input_dense)[0]

        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
