# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 20:58:33 2016

@author: annekespeijers
"""


doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health."

doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]


from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from stop_words import get_stop_words
en_stop = get_stop_words('en')
from nltk.stem.porter import PorterStemmer
p_stemmer = PorterStemmer()

texts = list()

for doc in doc_set:
	raw = doc.lower()
	tokens = tokenizer.tokenize(raw)
	stopped_tokens = [i for i in tokens if not i in en_stop]
	new_texts = [p_stemmer.stem(i) for i in stopped_tokens]
	texts.append(new_texts) # create list of lists


from gensim import corpora, models

# give each unique token an id number
dictionary = corpora.Dictionary(texts)
print(dictionary.token2id)

# convert dictionary to bag of words - list of lists with tuples saying how many of each word are in each doc
corpus = [dictionary.doc2bow(text) for text in texts]

