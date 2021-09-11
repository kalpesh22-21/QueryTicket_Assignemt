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
    
    
    
