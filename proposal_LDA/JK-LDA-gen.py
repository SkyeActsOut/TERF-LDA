#############
# Skye Kychenthal
#
# Sources:
# https://yanlinc.medium.com/how-to-build-a-lda-topic-model-using-from-text-601cdcbfd3a6
# https://github.com/explosion/spaCy/issues/3967
#
# Fixes used (for future projects)
# https://stackoverflow.com/questions/52166914/msys2-mingw64-pip-vc-6-0-is-not-supported-by-this-module
# --> use py -m pip install #package#
#
# NOTE Tweaking of this algorithm will occur in two places
# one is where gridsearch is done (which finds the best LDA model)
# and two is in the vectorizer which decides what words will actually be fed into LDA




import time
import numpy as np
import pandas as pd
import re
from nltk.stem import SnowballStemmer
import spacy
import gensim

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

print('### LOADING JK ROWLING BLOG ###')

# Read the J.K. Rowling data
f = open("JK-TEXT.txt", "r", encoding='utf8')
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


# lemmatization with snowball
# NOT USED
# def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
#     texts_out = []
#     # SnowballStemmer is the library being used to join words instead of spacey which is what the tutorial recommends
#     # this is primarily to keep TERF, and other acronyms/words not common to normal English discussions haha
#     stemmer = SnowballStemmer('english')
#     for sent in texts:
#         texts_out.append(' '.join(stemmer.stem(token) for token in sent))

#     return texts_out

print('### LOADING SPACY LARGE ###')

nlp = spacy.load('en_core_web_trf', disable=['parser', 'ner'])

# adding words to spacy vocabulary
words = ["terf", 'gc', 'gender critical']
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

# lemmatization with spacy


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in [
                         '-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out


# Leminizing currently with only nouns and adjectives - verbs might be helpful? adverbs likely not helpful.
data_lemmatized = lemmatization(data_words, ['NOUN', 'ADJ'])

print(data_lemmatized)

print('### VECTORIZING DATA ###')

# Vectorizes the lemminzied data to be fed through sci-kit learn's LDA model
vectorizer = CountVectorizer(analyzer='word',
                             min_df=4,
                             stop_words='english',
                             lowercase=True,
                             token_pattern='[a-zA-Z0-9]{2,}',
                             max_features=50000
                             )
data_vectorized = vectorizer.fit_transform(data_lemmatized)

# print(data_vectorized)

print('### RUNNING GRIDSEARCH TO FIND BEST LDA ###')

# FIND THE BEST FIT LDA MODEL
# USING COMBINATIONS OF ALL FOLLOWING VARIABLES
num_topics = [3, 4, 5, 6]
decay = [0.5, 0.7, 0.9]
offset = [10, 25, 50]
iter_max = [5, 10, 15]

search_params = {'n_components': num_topics,
                 'learning_decay': decay,
                 'learning_offset': offset,
                 'max_iter': iter_max}  # Init the Model
lda = LatentDirichletAllocation(max_iter=5, learning_method='online',
                                learning_offset=50., random_state=0)  # Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)  # Do the Grid Search
model.fit(data_vectorized)
GridSearchCV(cv=None, error_score='raise',
             estimator=LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
                                                 evaluate_every=-1, learning_decay=0.7, learning_method=None,
                                                 learning_offset=10.0, max_doc_update_iter=100, max_iter=15,
                                                 mean_change_tol=0.001, n_components=5, n_jobs=1, perp_tol=0.1, random_state=None,
                                                 topic_word_prior=None, total_samples=1000000.0),
             n_jobs=1,
             param_grid={'n_topics': num_topics,
                         'learning_decay': decay,
                         'learning_offset': offset,
                         'max_iter': iter_max},
             pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
             scoring=None)

best_lda_model = model.best_estimator_
print("Best Model's Params: ", model.best_params_)  # lower better
print("Best Log Likelihood Score: ", model.best_score_)  # higher better
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))

# transforms the model into read-able outputs
lda_output = best_lda_model.transform(data_vectorized)

print('### SORTING TOPICS ###')

# transforms the lda_output into a pandas data-frame that can be read
topicnames = ['Topic' + str(i) for i in range(best_lda_model.n_components)]
df_topic_keywords = pd.DataFrame(best_lda_model.components_)
df_topic_keywords.columns = vectorizer.get_feature_names()
df_topic_keywords.index = topicnames
df_topic_keywords.head()

# organizes the data-frame into the top x words for each topic


def show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords


topic_keywords = show_topics(
    vectorizer=vectorizer, lda_model=best_lda_model, n_words=15)

df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i)
                             for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i)
                           for i in range(df_topic_keywords.shape[0])]

# stylizing the array


def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)


def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)


name = f'exports/JK_Rowling_Topics_{time.time()}'

print(f'### DONE! EXPORTING AS: {name}')

# exports a topics.csv file to ./exports/ with the variables for the number of topics
df_topic_keywords.to_csv(f'{name}.csv')