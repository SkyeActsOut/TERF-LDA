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
    "min_length": "3"

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
num_topics = [5]
decay = [0.5, 0.7, 0.9]
offset = [25]
iter_max = [3]

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
# print("Best Model's Params: ", model.best_params_)  # lower better
# print("Best Log Likelihood Score: ", model.best_score_)  # higher better
# print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))

fin_settings = { 
                 "vector_settings": vector_settings,
                 "params": model.best_params_,
                 "score": model.best_score_,
                 "perplexity": best_lda_model.perplexity(data_vectorized)
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

# stylizing the array


def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)


def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)


name1 = f'exports/TERF_LDA_RESULTS_{time.time()}'
name2 = f'exports/TERF_LDA_SETTINGS_{time.time()}'

print(f'### DONE! EXPORTING AS: {name1}')

# exports a topics.csv file to ./exports/ with the variables for the number of topics
df_topic_keywords.to_csv(f'{name1}.csv')

with open(name2, "w") as outfile:
    outfile.write(json.dumps( fin_settings, indent=4 ))