# Problem Statement
In this capstone project, we will build a classifier that can by analysing text in the incidents and classify incidents to right functional groups can help organizations to 
reduce the resolving time of the issue 
 
## Project Overview: 
* Used the IT tickets dataset from Kaggle containing description of tickets and Assigned group
* Data Preparation - Pre-processed the text using NLTK, Google Translator,Spacy etc.
* Data Sampling - Imbalance in the dataset groups was handled by resampling.
* Generated Word Embeddings Using ELMO. Also used Word2Vec and Glove Vectors.
* Used LSTM, BERT, FastText, XGB and SVM on the embbedings. Finalized The Bi-directional LSTM.
* Pickle the model
* Creating a Flask API Endpoint to classify desccriptions using the pretrained model.

## Code and Resources Used 
**Python Version:** 3.7  
**Packages:** Fasttext, pandas, numpy, sklearn, matplotlib, flask, flask_bootstrap, pickle, tensorflow, XGB.
**For Web Framework Requirements:**  ```pip install -r requirements.txt```   


## EDA
* There are 6 Assignment Group’s for which just have 1 ticket in the dataset
* 15 Assignment groups have more than 100 tickets. 
* Only 20% of the Assignmentgroups have greater than 100 tickets.
* Missing values in Short description and Description columns.
* Presence of text other than english.
<img target="_blank" src="https://github.com/kalpesh22-21/IT_Ticket_Classification/blob/main/Pie%20chart.png" width=270>

### Top 20 Groups
<img target="_blank" src="https://github.com/kalpesh22-21/IT_Ticket_Classification/blob/main/Top%2020%20groups.png" width=600>

### Word Cloud for Group_0
<img target="_blank" src="https://github.com/kalpesh22-21/IT_Ticket_Classification/blob/main/Word%20Cloud%20for%20Group%200.png" width=270>


## Model performance
The Bi-directional LSTM with GLove Embeddings outperformed the other approaches on the test. 
<img target="_blank" src="https://github.com/kalpesh22-21/IT_Ticket_Classification/blob/main/Scores.png" width=400>

<img target="_blank" src="https://github.com/kalpesh22-21/IT_Ticket_Classification/blob/main/Comparison.png" width=700>


## Productionization 
In this step, I built a flask API endpoint that was hosted on a local webserver. The comments were then passed to complete pipeline where the text wast traslated if not in english,cleaned, tokenized and parsed though the model to evaluate the result. 

### Home.html

<img target="_blank" src="https://github.com/kalpesh22-21/QueryTicket_Classification/blob/main/Front%20End.png" width=700>
