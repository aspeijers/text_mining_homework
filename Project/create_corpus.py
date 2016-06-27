import numpy as np
import codecs
import nltk
import math
from nltk.tokenize import wordpunct_tokenize
from nltk import PorterStemmer
import pandas as pd
import json
import os
import dill as pickle

class Document():
    
    """ 
    The Document class represents a class of individul documents
    """
    
    def __init__(self, article_title, article_date, article_comments, article_text):
        self.title = article_title
        self.date = article_date
        self.comments = article_comments
        self.text = article_text.lower()
        self.tokens = np.array(wordpunct_tokenize(self.text))
        
        
    def token_clean(self,length):
        """ 
        description: strip out non-alpha tokens and tokens of length > 'length'
        input: length: cut off length 
        """

        self.tokens = np.array([t for t in self.tokens if (t.isalpha() and len(t) > length)])


    def stopword_remove(self, stopwords):
        """
        description: Remove stopwords from tokens.
        input: stopwords: a suitable list of stopwords
        """

        self.tokens = np.array([t for t in self.tokens if t not in stopwords])


    def stem(self, length):
        """
        description: Stem tokens with Porter Stemmer.
        """

        self.tokens = np.array([PorterStemmer().stem(t) for t in self.tokens])
        self.tokens = np.array([t for t in self.tokens if len(t) > length])


    
    def word_match(self, wordset):
        """
        description: return word count for a given set of words as a dictionary
        input: set of words (or possibly stemmed version of words) as a list or tuple
        """
        
        word_dict = {}
        for word in wordset:
            word_dict[word] = 0

        for word in self.tokens:
            if word_dict.get(word) != None:
                 word_dict[word] = word_dict[word] + 1
        return(word_dict)
 


##########################################################################################

class Corpus():
    
    """ 
    The Corpus class represents a document collection
     
    """
    def __init__(self, doc_data, stopword_file, clean_length):
        """
        Notice that the __init__ method is invoked everytime an object of the class
        is instantiated
        """
        
        # initialise documents by invoking the appropriate class
        self.docs = [Document(doc[0], doc[1], doc[2], doc[3]) for doc in doc_data] 
        
        # number of documents
        self.N = len(self.docs)
        self.clean_length = clean_length
        
        # get a list of stopwords
        self.create_stopwords(stopword_file, clean_length)  

        # stopword removal, token cleaning and stemming to docs
        self.clean_docs(clean_length)      
        
        # create vocabulary
        self.corpus_tokens()
        
        # create id dictionary for vocabulary
        self.token_id_dict()
     

    def create_stopwords(self, stopword_file, length):
        """
        description: parses a file of stopwords, removes words of length > 'length' and 
        stems it
        input: length: cutoff length for words
               stopword_file: stopwords file to parse
        """
        
        with codecs.open(stopword_file,'r','utf-8') as f: 
            raw = f.read()
        
        #self.stopwords = (np.array([PorterStemmer().stem(word) 
                                    #for word in list(raw.splitlines()) if len(word) > length]))
        self.stopwords = (np.array([word for word in list(raw.splitlines()) if len(word) > length]))
        

    def clean_docs(self, length):
        """ 
        Applies stopword removal, token cleaning and stemming to docs
        """
        
        for doc in self.docs:
            doc.token_clean(length)
            doc.stopword_remove(self.stopwords) 
            doc.stem(length) 
    
     
    def corpus_tokens(self):
        """
        description: create a set of all all tokens (the vocabulary) across entire corpus
        """
        
        #initialise an empty set
        self.token_set = set()
        for doc in self.docs:
            self.token_set = self.token_set.union(doc.tokens)
        
        # convert to list to set the order
        self.token_set = list(self.token_set)

    
    def token_id_dict(self):
        """
        description: creates and id for each token in the corpus tokens
        """
        
        self.token_dict = {key: value for (key,value) in zip(self.token_set,range(len(self.token_set)))}
    
    
    def document_term_matrix(self,wordset):
        """
        description: create a D by V array of frequency counts 
        note: order of both documents and words of the input are retained
        input: set of words (or possibly stemmed version of words) as a list or tuple. 
                can use the corpus.token_set. 
        """
        
        D = self.N
        V = len(wordset)
        doc_term_matrix = np.empty([D,V])
        
        for doc, i  in zip(self.docs, range(len(self.docs))) :
            # for each document, return the word count for the given wordset, as a dictionary 
            worddict = doc.word_match(wordset)
            
            for word, j in zip(wordset, range(len(wordset))):
                doc_term_matrix[i,j] = worddict[word]
        return(doc_term_matrix)
    
    
    
    def tf_idf(self,wordset):
        """
        description: create a D by V tf_idf array
        note: order of both documents and words of the input are retained
        input: set of words (or possibly stemmed version of words) as a list or tuple
        """
        
        matrix = self.document_term_matrix(wordset)      
        
        # number of documents that contain term v
        nonzero_matrix = matrix > 0 
        df = (nonzero_matrix).sum(axis = 0) 
        
        # inverse document frequency (set inf values to 0)
        idf = np.log( self.N / df )
        idf[np.isinf(idf)] = 0
        
        # term frequency of v in d        
        tf = (1 + np.log(matrix))
        tf[np.isinf(tf) ] = 0
        
        return(tf*idf)
    
    
    def dict_rank(self, wordset, freq_rep = True ):
        """
        description: returns documents title, date, number of comments and 
        score (in term of frequency count or tf-idf count), sorted by score, 
        for a given wordlist.
        output: array of document attributes as discussed in description, sorted.
        input: wordset - list (or tuple) of stemmed words
               freq_rep - True means we use document_term_matrix, tf_idf otherwise
        """
        
        if freq_rep:
            # use document term matrix (DxV np array)
            matrix = self.document_term_matrix(wordset)
        else:
            # use tf_idf matrix (DxV np array)
            matrix = self.tf_idf(wordset)
        
        # sum up the counts for each document (produces a 1 column np array, length D)
        matrix_total = matrix.sum(axis = 1)
        
        # title, date, comments, score. ie a list of tuples
        doc_list = []
        for doc, i in zip(self.docs, range(len(self.docs))):
            doc_list.append((self.docs[i].title, self.docs[i].date, self.docs[i].comments, matrix_total[i]))
        
        # sort list by score (descending)
        sorted_doc_list = sorted(doc_list, key=lambda x: x[3], reverse=True)
        return(sorted_doc_list)
        
        

    def weighted_dict_rank(self, sentiment_dict, freq_rep = True ):
        """
        description: Uses an imported weighted dictionary to return a sentiment score for each document
        output: list of year, president, sentiment score tuples sorted by sentiment score
        input: sentiment_dict - dictionary of words (or possibly stemmed version of words) with sentiment scores
        freq_rep - True means we use document_term_matrix, tf_idf otherwise
        """
       
        wordset = sentiment_dict.keys()
        values = sentiment_dict.values()
        
        if freq_rep:
            matrix = self.document_term_matrix(wordset)
        else:
            matrix = self.tf_idf(wordset)
        
        # multiply values in matrix by valence score of each word in sentiment_dict, Gives a Dx1 np array     
        matrix_total = np.dot(matrix, values)
        
        # title, date, comments, score (sorted by score). ie a list of tuples
        doc_list = []        
        for doc, i in zip(self.docs, range(len(self.docs))):
            doc_list.append((self.docs[i].title, self.docs[i].date, self.docs[i].comments, matrix_total[i]))
        
        # sort list by sentiment score (descending)
        sorted_doc_list = sorted(doc_list, key=lambda x: x[3], reverse=True)
        return(sorted_doc_list)
        

      
###################################################################################################
# set working directory
os.chdir("/Users/annekespeijers/Desktop/BGSE/Term3/TextMining/Homework/Project/")

# load in data
with open('./data/articles.txt') as file:
    #articles = json.load(file, encoding="utf-8")
    articles = json.load(file)

# create corpus from articles list
corpus = Corpus(articles, 'stopwords.txt', 2)

# save corpus object
with open('./data/corpus.pkl', 'wb') as output:
    pickle.dump(corpus, output, -1)

# # testing read in
# with open('./data/corpus.pkl', 'rb') as input:
#     corpus_test = pickle.load(input)




