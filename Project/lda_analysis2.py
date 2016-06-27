import os
import dill as pickle
import numpy as np
import lda
# import json
# from stop_words import get_stop_words
# from nltk.stem.porter import PorterStemmer
# from nltk.tokenize import wordpunct_tokenize

# set working directory
os.chdir("/Users/annekespeijers/Desktop/BGSE/Term3/TextMining/Homework/Project/")

# load in corpus
with open('corpus.pkl', 'rb') as input:
    corpus = pickle.load(input)

# doc term matrix
X = corpus.document_term_matrix(corpus.token_set)
X = X.astype(int)

# get vocab, article titles and article comments
vocab = tuple(corpus.token_set)
titles = tuple([t.title for t in corpus.docs])
comments = tuple([com.comments for com in corpus.docs])

# fit the model: k = 3
model = lda.LDA(n_topics=3, n_iter=500, random_state=1) # change params here
model.fit(X)

# topic - word
topic_word = model.topic_word_

# print top 5 words for each topic
n = 5
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
    print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))

# document - topic
doc_topic = model.doc_topic_

# from this I can get the topic that makes up the biggest part of the doc
# and plot topics against no of comments (perhaps take out ones that have less than 50% max)
