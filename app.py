# Core package
import streamlit as st
import altair as alt
# EDA
import pandas as pd
import numpy as np

# Templates
#HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius:0.25rem; padding: 1rem">{}</div"""

# Utils
import joblib
import os, base64
import time
import qrcode
from PIL import Image
import random
import string

#models
import spacy
#from spacy import displacy
nlp = spacy.load('en_core_web_sm')
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.summarization import summarize
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest


# helping functions

# Emotion Detection
pplnLR = joblib.load(open("models/emotion_classifier_final", "rb"))

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱",
                       "happy":"🤗", "joy":"😂", "neutral":"😐", 
                       "sadness":"😔", "shame":"😳", "surprise":"😮"}

def predict_emotions(docx):
    results = pplnLR.predict([docx])
    return results[0]

def get_prediction_prob(docx):
    results = pplnLR.predict_proba([docx])
    return results


# Text Summarization
def summarizer(text, manual, gensim, word_count):
    if manual:
        words_num = int(len(text.split()))
        stopwords = list(STOP_WORDS)
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
        #tokens = [token.text for token in doc]
        #punctuation = punctuation + '\n'
        
        word_frequencies = {}
        for word in doc:
            if word.text.lower() not in stopwords:
                if word.text.lower() not in punctuation:
                    if word.text not in word_frequencies.keys():
                        word_frequencies[word.text] = 1
                    else:
                        word_frequencies[word.text] += 1
        
        max_freq = max(word_frequencies.values())
        
        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word]/max_freq
            
        sentence_tokens = [sent for sent in doc.sents]
        
        sentence_scores={}
        for sent in sentence_tokens:
            for word in sent:
                if word.text.lower() in word_frequencies.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]
        
        select_length = int(len(sentence_scores)*(word_count/words_num))
        result = nlargest(select_length, sentence_scores, key= sentence_scores.get)                
                                                
    elif gensim:
        result = summarize(text, word_count=word_count)
    
    return result

# Document Redaction
def redact_terms(text, hide_term):
    doc = nlp(text)
    redacted = []
    with doc.retokenize() as retokenizer:
        for entity in doc.ents:
            retokenizer.merge(entity)
    for token in doc:
        if token.ent_type_ == 'PERSON' and 'names' in hide_term:
            redacted.append("[REDACTED NAME]")
        elif token.ent_type_ == 'GPE' and 'places' in hide_term:
            redacted.append("[REDACTED PLACE]")
        elif token.ent_type_ == 'DATE' and 'dates' in hide_term:
            redacted.append("[REDACTED DATE]")
        elif token.ent_type_ == 'DATE' and 'org' in hide_term:
            redacted.append("[REDACTED]")
        else: 
            redacted.append(token.text)
    return " ".join(redacted)

# Write to file
timestamp = time.strftime("%m-%d-%Y %H%M%p")
filename = 'myDoc_' + timestamp + '.txt'
def writetofile(text, filename):
    with open(os.path.join("downloads", filename), "w") as f:
        f.write(text)
    return filename

# Download files
def download_file(filename):
    readfile = open(os.path.join("downloads", filename)).read()
    b64 = base64.b64encode(readfile.encode()).decode()
    href = '<a href="data:file/readfile:base64,{}">Download File</a>(right click to save as file name)'.format(b64)
    return href

# QR-Code instantiate
qr = qrcode.QRCode(version=1, 
                   error_correction=qrcode.constants.ERROR_CORRECT_L,
                   box_size=10, border=14)

# Load QR-Code image
def load_QRCode(img):
    image = Image.open(img)
    return image

# Password Generator
def generate_password(characters, length):
    random.shuffle(characters)
    password=[]
    for i in range(length):
        password.append(random.choice(characters))
    random.shuffle(password)

    return "".join(password)

ALL_CHARACTERS = list(string.ascii_letters + string.digits + "!@#$%^&*()_+-=")
ALPHANUMERICS = list(string.ascii_letters + string.digits)
ALPHABETS = list(string.ascii_letters)
print(ALPHABETS)
# =============================================================================
# @st.cache
# def render_entities(text):
#     doc = nlp(text)
#     html = displacy.render(doc, style='ent')
#     html = html.replace("\n\n", "\n")
#     result = HTML_WRAPPER.format(html)
#     return result
# =============================================================================


def main():
    st.title('NLP App')
    menu = ["Emotion Analysis", 
            "Text Summarization", 
            "Document Redactor", 
            "QR Code Generator", 
            "Password Generator",
            "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == 'Emotion Analysis':
        st.subheader('Home-Emotion in text')
        
        with st.form(key='emotion'):
            raw_text = st.text_area("Type here")
            submit_text = st.form_submit_button(label='Submit')
            
        if submit_text:
            col1,col2 = st.columns(2)
            
            # Apply functions
            prediction = predict_emotions(raw_text)
            probability = get_prediction_prob(raw_text)
            
            with col1:
                st.success("Original Text")
                st.write(raw_text)
                
                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction, emoji_icon))
                st.write("Confidence: {}%".format(100* round(np.max(probability),2)))
                
                
            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, 
                                        columns=pplnLR.classes_)
                proba_df=proba_df.T.reset_index()
                proba_df.columns = ["Emotions","Probability"]
                #st.write(proba_df)
                
                fig = alt.Chart(proba_df).mark_bar().encode(x='Emotions',
                                                            y='Probability',
                                                            color='Emotions')
                
                st.altair_chart(fig,use_container_width=True)
                
                
        
    elif choice == 'Text Summarization':
        st.subheader('Text Summarization')
        
        
        raw_text = st.text_area("Input Text For Summary",height=300)
        summary_options = st.selectbox("Choose Summarizer",['manual','gensim'])
        text_range= st.sidebar.slider("Summarize words Range", 50, 250)
                                      #int(len(raw_text.split()) * 0.1),
                                      #int(len(raw_text.split()) * 0.25))
        if st.button("summarize"):
            if summary_options == 'gensim': 
                st.success(summarizer(raw_text, False, True, 
                                      word_count=text_range))
            else:
                st.success(summarizer(raw_text, True, False, 
                                      word_count=text_range))
        
    elif choice == 'Document Redactor':
        st.subheader('Document Redactor')
        raw_text = st.text_area("Input Text For Redaction",height=300)
        redaction_type = ["names", "places", "org", "dates"]
        hide_term = st.multiselect("Censor terms:", redaction_type)
        save = st.radio("Save the file?", ('Yes','No'))
        if st.button("Submit"):
            if save == 'Yes':
                new_text = redact_terms(raw_text, hide_term) 
                #st.subheader("Original Text")
                #st.write(render_entities(raw_text))
                
                st.subheader("Redacted Text")
                st.write(new_text)
        
                file_output = writetofile(new_text, filename)
                st.info("Saved Result As: {}".format(filename))
                download_link = download_file(file_output)
                st.markdown(download_link, unsafe_allow_html=True)
        
        
    elif choice == "QR Code Generator":
        st.subheader('QR-Code Generator')
        with st.form(key='QR-code'):
            raw_text = st.text_area("Input Text Here",height=100)
            submit = st.form_submit_button("Generate QR-Code")
            
            
            if submit:
                col1, col2 = st.columns(2)
                
                with col1:
                    qr.add_data(raw_text)
                    qr.make(fit=True)
                    
                    img = qr.make_image(fill_color='black', back_color='white')
                    filename_img = 'generate_image_{}.png'.format(timestamp)
                    img_path = os.path.join('images', filename_img)
                    img.save(img_path)
                    
                    loaded_image = load_QRCode(img_path)
                    st.image(loaded_image)
                    
                with col2:
                    st.info("Original Text")
                    st.write(raw_text)
        
        
    elif choice == "Password Generator":
        st.subheader("Password Generator")
        pass_type = ['alphabet', 'alpha-numeric', 'all']
        pass_len = st.number_input('Password length', min_value=5, max_value=40, value=6)
        pass_choice = st.selectbox("Pattern", pass_type)
        if st.button("Generate"):
            st.info('Generate Password')

            if pass_choice == 'alphabet':
                generated_password = generate_password(ALPHABETS, pass_len)
                st.write(generated_password)
            elif pass_choice == 'alpha-numeric':
                generated_password = generate_password(ALPHANUMERICS, pass_len)
                st.write(generated_password)
            else:
                generated_password = generate_password(ALL_CHARACTERS, pass_len)
                st.write(generated_password)


            

    else:
        st.subheader('About')
        st.info("Georgopoulos Spyros - Data Scientist/Physicist")
    
     
if __name__ == '__main__':
    main()