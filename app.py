import streamlit as st
from joblib import load
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer

clf = load('filename.joblib')

vocabulary = joblib.load('vectorizer')
cv = CountVectorizer(vocabulary=vocabulary)


st.title("SENTIMENT ANALYSIS")
image = Image.open('IMG-20200904-WA0251.jpg')
st.image(image, width=800)
review = st.text_input('Enter your short review :')
df = {'review': review}
df = pd.DataFrame(df, index=[0])
to_pred = df.iloc[:, 0]
result = clf.predict(cv.transform(to_pred))
if(st.button('Predict')):
    st.write(result[0])
    
    
