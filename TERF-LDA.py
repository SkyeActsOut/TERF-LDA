"""
AUTHOR: Skye Kychenthal

NOTE Sources:
     https://yanlinc.medium.com/how-to-build-a-lda-topic-model-using-from-text-601cdcbfd3a6
     https://github.com/explosion/spaCy/issues/3967

NOTE Fixes used (for future projects)
     https://stackoverflow.com/questions/52166914/msys2-mingw64-pip-vc-6-0-is-not-supported-by-this-module
     --> use py -m pip install #package#

NOTE Tweaking of this algorithm will occur in two places
     one is where gridsearch is done (which finds the best LDA model)
     and two is in the vectorizer which decides what words will actually be fed into LDA
"""

import time, datetime

_t_start = datetime.datetime.now()

import json
import numpy as np
import pandas as pd

from CleanCSV import CSV_Data

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

print ("### LOADING CLEAN DATA ###")

data = CSV_Data('clean_text_dirty.csv')

print('### VECTORIZING DATA ###')

# Vectorizes the lemminzied data to be fed through sci-kit learn's LDA model
vector_settings = {

    "min_df": 15,
    "min_length": "4"

}

print (vector_settings)

vectorizer = CountVectorizer(analyzer='word',
                             min_df=vector_settings['min_df'],
                             stop_words='english',
                             lowercase=True,
                             token_pattern='[a-zA-Z0-9]{' + vector_settings['min_length'] + ',}',
                             max_features=50000
                             )
data_vectorized = vectorizer.fit_transform(data.get())

# print(data_vectorized)

print('### RUNNING GRIDSEARCH TO FIND BEST LDA ###')

# FIND THE BEST FIT LDA MODEL
# USING COMBINATIONS OF ALL FOLLOWING VARIABLES
num_topics = [12]
decay = [0.8]
"""
NOTE
learning_decay : float, default=0.7
It is a parameter that control learning rate in the online learning
method. The value should be set between (0.5, 1.0] to guarantee
asymptotic convergence. When the value is 0.0 and batch_size is
``n_samples``, the update method is same as batch learning. In the
literature, this is called kappa.
https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_lda.py
"""

offset = [100, 115, 120, 125, 130]
""" 
NOTE
learning_offset : float, default=10.0
A (positive) parameter that downweights early iterations in online
learning.  It should be greater than 1.0. In the literature, this is
called tau_0.
"""

iter_max = [100]
"""
NOTE
max_iter : int, default=10
The maximum number of passes over the training data (aka epochs).
It only impacts the behavior in the :meth:`fit` method, and not the
:meth:`partial_fit` method.
"""

batch_size=[256]
""" 
NOTE
batch_size : int, default=128
Number of documents to use in each EM iteration. Only used in online
learning.
"""

all_params = {

    "num_topics": num_topics,
    "decay": decay,
    "offset": offset,
    "iter_max": iter_max,
    "batch_size": batch_size

} 

search_params = {'n_components': num_topics,
                 'learning_decay': decay,
                 'learning_offset': offset,
                 'max_iter': iter_max,
                 'batch_size': batch_size}  # Init the Model
lda = LatentDirichletAllocation(learning_method='online', random_state=0)  # Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params, verbose=2, n_jobs=5)  # Do the Grid Search
model.fit(data_vectorized)

best_lda_model = model.best_estimator_

# Higher score, lower perplexity is good
fin_settings = { 
                 "vector_settings": vector_settings,
                 "params": model.best_params_,
                 "score": model.best_score_,
                 "perplexity": best_lda_model.perplexity(data_vectorized),
                 "all_settings": all_params
                }
print (fin_settings)

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

name1 = f'exports/TERF_LDA_RESULTS_{time.time()}'

print(f'### DONE! EXPORTING AS: {name1}')

_t_est = datetime.datetime.now() - _t_start
fin_settings['time_in_mins'] = _t_est.total_seconds()/60
# fin_settings['num_runs'] = n_of_runs

# fin_settings['csv'] = df_topic_keywords.to_json()

# exports a topics.csv file to ./exports/ with the variables for the number of topics
df_topic_keywords.to_csv(f'{name1}.csv')

with open("ALL_SETTINGS.json", "r+") as outfile:

    f = json.load (outfile)

    f[name1] = fin_settings

    outfile.seek(0)

    json.dump( f, outfile, indent=4 )