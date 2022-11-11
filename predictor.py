import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from CleanCSV import CSV_Data

import pickle
pickled_model = pickle.load(open('./broad_model.pk1', 'rb'))

import time, datetime
import pandas as pd
import re
import spacy
import gensim
import copy
import glob
import csv

from CleanText import sent_to_words, lemmatization, nlp

# Re-create the vectorizer
vector_settings = {

    "min_df": 15,
    "min_length": "4"

}
data = CSV_Data('clean_text_dirty.csv')
vectorizer = CountVectorizer(analyzer='word',
                             min_df=vector_settings['min_df'],
                             stop_words='english',
                             lowercase=True,
                             token_pattern='[a-zA-Z0-9]{' + vector_settings['min_length'] + ',}',
                             max_features=50000
                             )
vectorizer.fit_transform(data.get())

df_topic_keywords = pd.read_csv('./exports/TERF_LDA_RESULTS_1668123536.3682024.csv')

def predict_topic(text, nlp=nlp):
    global sent_to_words
    global lemmatization# Step 1: Clean with simple_preprocess
    mytext_2 = list(sent_to_words(text))# Step 2: Lemmatize
    mytext_3 = lemmatization(mytext_2, allowed_postags=['NOUN', 'ADJ', 'VERB'])# Step 3: Vectorize transform
    mytext_4 = vectorizer.transform(mytext_3)# Step 4: LDA Transform
    topic_probability_scores = pickled_model.transform(mytext_4)
    _topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), 1:14]
    topic = f"Topic {_topic.name+1}"
    
    # Step 5: Infer Topic
    infer_topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), -1]
    
    #topic_guess = df_topic_keywords.iloc[np.argmax(topic_probability_scores), Topics]
    return infer_topic, topic, topic_probability_scores# Predict the topic

# mytext = ["Another supporter comment from our legal crowdfunder:Defending the right to be LGB and same sex attracted isnt something I thought we would have to be fighting for in 2022Please help us defend our charitable statuse"]
# infer_topic, topic, prob_scores = predict_topic(text = mytext)

files = [ open(f, errors='ignore') for f in glob.glob("tweets/*.csv")]

for f in files:

    likes_total = 0

    new = copy.copy(list(csv.reader(f)))

    print (f'### reading {f.name} ###')

    for line in new:

        if (len(line) == 0): 
            continue

        if (line[1] == True or line[2] == True): #if is QRT or RT
            continue
    
        likes = line[5]

        if (likes == 'Likes'):  
            continue

        text = line[3]

        infer_topic, topic, prob_scores = predict_topic(text = [text])

        line.append(topic)

    print (f'### DONE W {f.name} ###')

    fname = f.name.replace('tweets', 'tweets_w_topic')
    with open (fname, 'w') as _f:

        writer = csv.writer(_f)

        writer.writerows(new)