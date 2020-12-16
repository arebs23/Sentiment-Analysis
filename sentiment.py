import streamlit as st
import pandas as pd
from preprocessing import *
import pickle
import spacy
import spacy_streamlit
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt






def make_wordcloud(new_text):
    ''''funciton to make wordcloud'''
    
    wordcloud = WordCloud(width = 800, height = 800, 
                min_font_size = 10,
                background_color='black', 
                colormap='Set2', 
                collocations=False).generate(new_text) 
    
    #wordcloud.recolor(color_func = grey_color_func)

    
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.gca().imshow(wordcloud, interpolation='bilinear')
    
    plt.gca().axis("off") 
    plt.tight_layout(pad = 0) 

    st.pyplot()


#@st.cache(allow_output_mutation = True)

clean_text = CleanText()
tc = TextCounts()

model = pickle.load(open('sent_model.pkl','rb'))

models = ["en_core_web_sm", "en_core_web_md"]

st.title("NLP with Streamlit")
st.subheader("Natural Language Processing on the Go")
activities = ["Sentiment Analysis","NER Checker","word cloud"]
choice = st.sidebar.selectbox("Select Activities",activities)


if choice == 'Sentiment Analysis':
    st.subheader("Sentiment of your text")
    message = st.text_area("Enter your text","Type Here")
      
      
      
    message = pd.Series(message)
    df_counts_pos = tc.transform(message)
    df_clean_pos = clean_text.transform(message)
    df_model_pos = df_counts_pos
    df_model_pos['clean_text'] = df_clean_pos


    if st.button("Predict"):
        prediction = model.predict(df_model_pos)
        st.write(f'sentiment prediction is {prediction[0]}')
    
if choice == 'NER Checker' :
    st.subheader("Entity recognition of your text")
    message = st.text_area("Enter your text","Type Here")
    message = pd.Series(message)
    message = clean_text.fit_transform(message)


    if st.button("Analyze"):
        spacy_streamlit.visualize(models, message)

if choice == 'word cloud':
    st.subheader("Word cloud display")
    
    message = st.text_area("Enter your text","Type Here")
    message = pd.Series(message)
    make_wordcloud(' '.join(message))
        


