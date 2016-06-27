import os
import dill as pickle
import numpy as np
import lda
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import IntVector, FloatVector

# set working directory
os.chdir("/Users/annekespeijers/Desktop/BGSE/Term3/TextMining/Homework/Project/")

# load in corpus
with open('./data/corpus.pkl', 'rb') as input:
    corpus = pickle.load(input)

# doc term matrix
X = corpus.document_term_matrix(corpus.token_set)
X = X.astype(int)

# get vocab, article titles and article comments
vocab = tuple(corpus.token_set)
titles = tuple([t.title for t in corpus.docs])
comments = tuple([com.comments for com in corpus.docs])


## fit the model: k = 2
k = 2
model = lda.LDA(n_topics=k, n_iter=500, random_state=1, eta=200/float(len(vocab)), alpha=50/float(k) ) 
model.fit(X)

# topic - word
topic_word = model.topic_word_

# print top 5 words for each topic
n = 5
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
    print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))


## fit the model: k = 3
k = 3
model = lda.LDA(n_topics=k, n_iter=500, random_state=1, eta=200/float(len(vocab)), alpha=50/float(k) )
model.fit(X)

# topic - word
topic_word = model.topic_word_

# print top 5 words for each topic
n = 5
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
    print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))


## fit the model: k = 4
k = 4
model = lda.LDA(n_topics=k, n_iter=500, random_state=1, eta=200/float(len(vocab)), alpha=50/float(k) )
model.fit(X)

# topic - word
topic_word = model.topic_word_

# print top 5 words for each topic
n = 5
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
    print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))


## fit the model: k = 5
k = 5
model = lda.LDA(n_topics=k, n_iter=500, random_state=1, eta=200/float(len(vocab)), alpha=50/float(k) )
model.fit(X)

# topic - word
topic_word = model.topic_word_

# print top 5 words for each topic
n = 5
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
    print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))


## fit the model: k = 6
k = 6
model = lda.LDA(n_topics=k, n_iter=500, random_state=1, eta=200/float(len(vocab)), alpha=50/float(k) )
model.fit(X)

# topic - word
topic_word = model.topic_word_

# print top 5 words for each topic
n = 5
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
    print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))


## Choose 3 topics
k = 3
model = lda.LDA(n_topics=k, n_iter=500, random_state=1, eta=200/float(len(vocab)), alpha=50/float(k) )
model.fit(X)

# topic - word
topic_word = model.topic_word_

# document - topic
doc_topic = model.doc_topic_

# calculate comments assigned to each topic
topic_comments = np.dot(comments, doc_topic)

## plot results

# import r devices
#base = importr('base')
rbarplot = robjects.r('barplot')
#rprint = robjects.globalenv.get("print")
#graphics = importr("graphics")

# plots
grdevices.png("./plots/topic_comments.png")
rbarplot(robjects.FloatVector(topic_comments), xlab="Topics", ylab="Comments", main="Comments assigned to each topic", col="coral1")
grdevices.dev_off()



# Generate plots for other values of k
###
k = 4
model = lda.LDA(n_topics=k, n_iter=500, random_state=1, eta=200/float(len(vocab)), alpha=50/float(k) )
model.fit(X)
topic_word = model.topic_word_
doc_topic = model.doc_topic_
topic_comments = np.dot(comments, doc_topic)

grdevices.png("./plots/topic_comments4.png")
rbarplot(robjects.FloatVector(topic_comments), xlab="Topics", ylab="Comments", main="Comments assigned to each topic (k = 4)", col="coral1")
grdevices.dev_off()

###
k = 5
model = lda.LDA(n_topics=k, n_iter=500, random_state=1, eta=200/float(len(vocab)), alpha=50/float(k) )
model.fit(X)
topic_word = model.topic_word_
doc_topic = model.doc_topic_
topic_comments = np.dot(comments, doc_topic)

grdevices.png("./plots/topic_comments5.png")
rbarplot(robjects.FloatVector(topic_comments), xlab="Topics", ylab="Comments", main="Comments assigned to each topic (k = 5)", col="coral1")
grdevices.dev_off()

###
k = 6
model = lda.LDA(n_topics=k, n_iter=500, random_state=1, eta=200/float(len(vocab)), alpha=50/float(k) )
model.fit(X)
topic_word = model.topic_word_
doc_topic = model.doc_topic_
topic_comments = np.dot(comments, doc_topic)

grdevices.png("./plots/topic_comments6.png")
rbarplot(robjects.FloatVector(topic_comments), xlab="Topics", ylab="Comments", main="Comments assigned to each topic (k = 6)", col="coral1")
grdevices.dev_off()

