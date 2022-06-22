# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:25:26 2022

@author: BIPIN
"""
from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nlp
import streamlit as st

dataset = nlp.load_dataset('emotion')
train = dataset['train']

def get_tweet(data):
  tweets = [x['text'] for x in data]
  labels = [x['label'] for x in data]
  return tweets, labels

tweets,labels = get_tweet(train)

tokenizer = Tokenizer(num_words=100000, oov_token='<UNK>')
tokenizer.fit_on_texts(tweets)

maxlen = 200
classes = set(labels)
class_to_index = dict((c,i) for i,c in enumerate(classes))
index_to_class = dict((v,k) for k , v in class_to_index.items())

model = load_model('IR.h5')

st.title("Emotions Classification of General Text")
st.markdown("This application is a dashboard that is used to do classfication of text/comments/tweets into anger😠, joy😂, fear😨, sadness😞, surprise😯 or love💓 ") 
st.markdown("##### Note: Input text/comment/tweets first and then switch to results/classification because without input text the variable will take undefined value")  
st.image('https://neurohive.io/wp-content/uploads/2018/12/1Rt4vf-ld0uJsJm3TwMuHfg-e1545648760934.jpeg')

with st.form(key="my_form"):
    i = st.text_input('Enter Your Text To Be Classified')
    submitted = st.form_submit_button("Submit")

if submitted:
    st.success("Your Text is stored Successfully")
    st.write("This is your Text: ", i)
    
        
a = tokenizer.texts_to_sequences([i])
b = pad_sequences(a, truncating='post', padding='post', maxlen = maxlen)
p = model.predict(b)
pred_class = index_to_class[np.argmax(p).astype('uint8')]

with st.form(key="my_form1"):
    st.write('Click Below For Result')
    sub = st.form_submit_button("Click Here")

    if sub:
         if pred_class == 'anger':
             st.markdown('##### Your Text/Comment/Tweet is classfied as ANGER')
             st.image('https://cdn.shopify.com/s/files/1/1061/1924/products/Super_Angry_Face_Emoji_ios10_large.png?v=1571606092', width=256)
         elif pred_class == 'joy':
             st.markdown('##### Your Text/Comment/Tweet is classfied as JOY')
             st.image('http://cdn.shopify.com/s/files/1/1061/1924/products/Tears_of_Joy_Emoji_8afc0e22-e3d4-4b07-be7f-77296331c687_grande.png?v=1571606036', width=256)
         elif pred_class == 'fear':
             st.markdown('##### Your Text/Comment/Tweet is classfied as FEAR')
             st.image('https://i.pinimg.com/originals/61/93/b6/6193b6d1347130f55a37da408df50670.png')    
         elif pred_class == 'sadness':
             st.markdown('##### Your Text/Comment/Tweet is classfied as SADNESS')
             st.image('https://www.cambridge.org/elt/blog/wp-content/uploads/2019/07/Sad-Face-Emoji-480x480.png')    
         elif pred_class == 'surprise':
             st.markdown('##### Your Text/Comment/Tweet is classfied as SURPRISE')
             st.image('https://cdn.shopify.com/s/files/1/1061/1924/products/Emoji_Icon_Surprised_with_teeth_large.png?v=1571606093')     
         else:
             st.markdown('##### Your Text/Comment/Tweet is classfied as LOVE')
             st.image('https://www.citypng.com/public/uploads/preview/red-heart-emoji-whatsapp-emoticon-love-11583747586pa1uapnfjn.png')     
        
        
        