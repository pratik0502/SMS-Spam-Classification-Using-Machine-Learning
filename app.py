import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize into words

    y = []  # To store words without special characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]  # Remove stopwords and punctuation
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]  # Perform stemming
    y.clear()

    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)  # Convert back to string from list of words


tfidf = pickle.load(open('vectorizer1.pkl','rb'))
model = pickle.load(open('model1.pkl','rb'))

st.title('Email/Sms Spam Classifier')

input_sms = st.text_input("Enter the Message")

if st.button('Predict'):

    # preprocessing

    transform_sms = transform_text(input_sms)

    # vectorize

    vector_input = tfidf.transform([transform_sms])

    # predict
    result = model.predict(vector_input)[0]

    # display

    if result == 1:
        st.header('SPAM')
    else:
        st.header('Not SPAM')
