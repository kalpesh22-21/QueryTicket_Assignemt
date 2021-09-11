import pickle
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import numpy as np
from dateutil import parser
from google_trans_new import google_translator
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import spacy
from tensorflow import keras
from tensorflow.math import reduce_mean
import re

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

app = Flask(__name__)
Bootstrap(app)

def lemmatize(text):
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
    model = keras.models.load_model('C:\\Users\\Kalpesh\\Great lakes\\Capstone\\model')
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
    f=open('C:\\Users\\Kalpesh\\Great lakes\\Capstone\\labels.txt','r')
    lines = f.readlines()
    Labels = []
    for line in lines:
        Labels.append(line.replace('\n',''))
    return(Labels)

@app.route('/')
def man():
    model = load_model()
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    maxlen = 50
    embedding_size = 300
    S_D = translate(Formatting(request.form['b']))
    D = translate(Formatting(request.form['c']))
    if str(S_D) == str(D):
        Desc = str(D)
    else:
        Desc = str(S_D) + str(D)
    Desc = " ".join(word for word in Desc.split(' ') if word not in stopwords.words('english'))
    Desc = lemmatize(Desc)
    tk = Tokenizer()
    with open('C:\\Users\\Kalpesh\\Great lakes\\Capstone\\tokenizer.pickle', 'rb') as handle:
        tk = pickle.load(handle)
    X = tk.texts_to_sequences(Desc)
    X = pad_sequences(X,maxlen=maxlen,padding='post')
    Labels = load_labels()
    model = load_model()
    pred = np.argmax(reduce_mean(model.predict(X),0))
    
    return render_template('after.html', data=Labels[pred])


if __name__ == "__main__":
    app.run(debug=True)
    
    
    
