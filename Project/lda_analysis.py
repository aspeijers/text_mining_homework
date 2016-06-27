################################## LSA ########################################
import gensim
import os
import json
import codecs
from nltk.tokenize import wordpunct_tokenize
#from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models


# set working directory
os.chdir("/Users/annekespeijers/Desktop/BGSE/Term3/TextMining/Homework/Project/")

# load in data
with open('articles.txt') as file:
    articles = json.load(file)



# define tokenizer
#tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()



# create nested list of stemmed/stopped tokens for each document
texts = []

for art in articles:
    
    # clean and tokenise the article string
    raw = art[3].lower()
    tokens = wordpunct_tokenize(raw)
    
    # remove stop words and any words with length of 1
    stopped_tokens = [i for i in tokens if i not in en_stop if len(i) > 2 if i.isalpha()]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add stemmed tokens to list    
    texts.append(stemmed_tokens)


# create term - id dictionary
dictionary = corpora.Dictionary(texts)

# calculate word count for each document 
#(nb. this is not a sparse matrix, only words with >0 count included)
# this is the "bag of words" (bow)
corpus1 = [dictionary.doc2bow(text) for text in texts]

# run LDA
ldamodel2 = gensim.models.ldamodel.LdaModel(corpus1, 
                                           num_topics = 3, 
                                           id2word = dictionary, 
                                           passes = 2,  # same results for 20
                                           chunksize = 101, # doesn't make a difference at 2000
                                           update_every = 1, 
                                           alpha='symmetric', 
                                           eta = None, 
                                           decay=0.5, 
                                           offset=1.0, 
                                           eval_every=10, 
                                           iterations=50, # same results for 100
                                           gamma_threshold=0.001, 
                                           minimum_probability=0.01)
perplex = ldamodel.bound(corpus1)
print(perplex)

print(ldamodel.print_topics(num_topics=3, num_words=5))

# get topic probability distribution for the first document
print(ldamodel[dictionary.doc2bow(texts[0])])

a=ldamodel.get_document_topics(corpus1[100], minimum_probability=0)
with codecs.open("stopwords.txt",'r','utf-8') as f: 
    raw = f.read()
stopwords_1=raw.splitlines()





import numpy as np 
import lda
import lda.datasets
X = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()
titles = lda.datasets.load_reuters_titles()
model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
model.fit(X)
model.components_
model.loglikelihood() 
topic_word = model.topic_word_
n_top_words = 8

for i, topic_dist in enumerate(topic_word):
    # rearrange vocab into least used to most used and take most 8 used for each topic
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

x = np.array([3, 1, 2,5,6,7,8])




