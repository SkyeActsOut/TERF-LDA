import time, datetime
import numpy as np
import pandas as pd
import re
import spacy
import gensim
import csv

print('### LOADING RAW TEXT ###')

# Read the J.K. Rowling data
f = open("tweets/RAW.txt", "r", encoding='utf8')
data = f.read()

print('### CLEANING SPEECH ###')

# cleans the speech for any characters that may cause errors
re.sub(r'\s+', ' ', data)         # clean new line
re.sub(r'\S*@\S*\s?', '', data)   # clean @
re.sub(r"\'", "", data)           # Clean single quotes
re.sub('-', ' ', data)            # clean -

print('### "GENSIM" SENTENCE SPLITTING ###')

# takes a sentence and splits it into individual words, removing any that are unimportant
# using the gensim library


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


# gets the words for each sentence using sent_to_words and creates a list
data_words = list(sent_to_words(data.split('.')))


# Lemminizes (or truncates each word to it's core parts so "say", "saying", "said" all truncate to "say" )

print('### LOADING SPACY LARGE ###')

nlp = spacy.load('en_core_web_trf', disable=['parser', 'ner'])

# adding words to spacy vocabulary
# specifically in regards to TERFs, etc.
words = ["terf", 'gc', 'gender critical', 'agp', 'tra', 'womyn', 'TIW', "TIM", 'radfem', 'wombyn', "LGBA", 'LGB', "SexNotGender", "Transgenderism"]
vector_data = {}
for w in words:
    vector_data[w] = np.random.uniform(-1, 1, (300,))

vocab = nlp.vocab  # Get the vocab from the model
for word, vector in vector_data.items():
    print(f'adding {word} to vocab')
    vocab.set_vector(word, vector)

if not nlp.vocab.has_vector('terf'):
    print("EXIT NO TERF")
    exit()

print('### LEMINIZING WITH SPACY ###')


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    _t_start = datetime.datetime.now()
    cnt = 0
    _l = len(texts)
    texts_out = []
    for sent in texts:
        cnt += 1

        # Time estimate for the process. Typically 20 to 30 minutes. Yikes. 
        if (cnt == 100):
            _t_est = datetime.datetime.now() - _t_start
            print (f"{cnt}/{100} UNTIL ESTIMATE")
            print (f"{_t_est.total_seconds()/60 * (_l / 100)} MINUTES UNTIL DONE")

        doc = nlp(" ".join(sent))
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in [
                         '-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out


# Leminizing currently with only nouns and adjectives - verbs might be helpful? adverbs likely not helpful.
data_lemmatized = lemmatization(data_words, ['NOUN', 'ADJ', 'VERB'])

with open('clean_text.csv', "w") as f:

    write = csv.writer(f)
    write.writerows(data_lemmatized)