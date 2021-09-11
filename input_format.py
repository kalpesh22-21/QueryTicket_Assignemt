import numpy as np
from dateutil import parser
from google_trans_new import google_translator
import spacy
from tensorflow import keras
import re


def lemmatize(text):
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    doc = nlp(text)
    allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']
    temp = []
    for word in doc:
         if word.pos_ in allowed_postags:
            temp.append(word.lemma_)
    return(' '.join(temp))



def translate(text):
    translator = google_translator(url_suffix="ind",timeout=5)
    if (translator.detect(text))[0] != 'en':
        translatedText = translator.translate(text)
        time.sleep(0.5)
        return translatedText.lower()
    else:
        return(text.lower())
    
    
def is_date(str_):
    try:
        parser.parse(str_)
        return True
    except:
        return False
        
def load_model():
    model = keras.models.load_model('model')
    return(model)

def Formatting(text):
    text = str(text)
    text=text.lower()
    # Removing date from the text
    text = ' '.join([w for w in text.split() if not is_date(w)])
    # Remove numbers 
    text = re.sub(r'\d+','' ,text)
    #Remove email 
    text = re.sub(r'\S*@\S*\s?', '', text)
    # Remove new line characters 
    text = re.sub(r'\n',' ',text)
    # Remove hashtag while keeping hashtag text
    text = re.sub(r'#','', text)
    #& 
    text = re.sub(r'&;?', 'and',text)
    # Remove HTML special entities (e.g. &amp;)
    text = re.sub(r'\&\w*;', '', text)
    # Remove hyperlinks
    text = re.sub(r'https?:\/\/.*\/\w*', '', text)  
    # Removing addressings
    text = re.sub(r"received from:",' ',text)
    text = re.sub(r"from:",' ',text)
    text = re.sub(r"to:",' ',text)
    text = re.sub(r"subject:",' ',text)
    text = re.sub(r"sent:",' ',text)
    text = re.sub(r"ic:",' ',text)
    text = re.sub(r"cc:",' ',text)
    text = re.sub(r"bcc:",' ',text)
    # Remove characters beyond Readable formart by Unicode:
    text= ''.join(charac for charac in text if charac <= '\uFFFF') 
    text = text.strip()
    # Remove unreadable characters  (also extra spaces)
    text = ' '.join(re.sub("[^\u0030-\u0039\u0041-\u005a\u0061-\u007a]", " ", text).split())
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    text = re.sub(' +', ' ', text)
    text = text.strip()
    return text
def load_labels ():
    f=open('labels.txt','r')
    lines = f.readlines()
    Labels = []
    for line in lines:
        Labels.append(line.replace('\n',''))
    return(Labels)
