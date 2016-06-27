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
        


##########################################################################################

def read_dictionary(path):
    '''
    Read in and format and stem dictionary
    output: list of stemmed words
    '''
    file_handle = open(path)
    file_content = file_handle.read()
    file_content = file_content.lower()
    stripped_text = re.sub(r'[^a-z\s]',"",file_content)
    stripped_text = stripped_text.split("\n")
    #remove the last entry
    del stripped_text[-1]
    # remove duplicates
    stripped_text = list(set(stripped_text))
    # we need to stem it
    stemmed = [PorterStemmer().stem(i) for i in stripped_text]
    return(stemmed)
    
    
def convert_to_dictionary(file):
        """
        convert the afinn text file to dictionary
        Note: all dictionary entries are lower case 
        Line_split is tab
        entries in the afinn list are stemmed and the average valence score is taken
        """
        #Open file with proper encoding
        with codecs.open(file,'r','utf-8') as f: row_list = [ line.split('\t') for line in f ]
        dict_elements = [ ( PorterStemmer().stem(word[0]) , int(word[1]) ) for word in row_list ]
        
        #Take unique elements after stemming
        #problem: we have words with the same root and diff. score - solution: average scores
        
        # turn into pandas dataframe
        dict_elements_df = pd.DataFrame(dict_elements, index = range(len(dict_elements)), columns=['stem', 'value']) #2477 entries
        
        # group by stemmed words and average
        grouped = dict_elements_df.groupby('stem', as_index=False)
        dict_elements_agg = grouped.aggregate(np.mean) #1482 entries
    
        # turn pandas df back into dictionary
        dict_stems_averaged = dict_elements_agg.set_index('stem')['value'].to_dict()
        
        return(dict_stems_averaged)    
        
###################################################################################################
# set working directory
os.chdir("/Users/annekespeijers/Desktop/BGSE/Term3/TextMining/Homework/Project/")

# load in data
with open('articles.txt') as file:
    #articles = json.load(file, encoding="utf-8")
    articles = json.load(file)

# create corpus from articles list
corpus = Corpus(articles, 'stopwords.txt', 2)

# save corpus object
with open('corpus.pkl', 'wb') as output:
    pickle.dump(corpus, output, -1)

# testing read in
with open('corpus.pkl', 'rb') as input:
    corpus_test = pickle.load(input)

    
# investigate distribution of document term matrix
#doc_term_matrix = corpus.document_term_matrix(corpus.token_set)

###############################################################################
# load libraries for plotting with R
from rpy2 import robjects
from rpy2.robjects import Formula
from rpy2.robjects.vectors import IntVector, FloatVector
from rpy2.robjects.lib import grid
from rpy2.robjects.packages import importr
import rpy2.robjects.pandas2ri

# The R 'print' function
rprint = robjects.globalenv.get("print")
rhist = robjects.r('hist')

stats = importr('stats')
grdevices = importr('grDevices')
base = importr('base')
ggplot2 = importr('ggplot2')
graphics = importr("graphics")

############################# Data Descriptives ###############################
# define term frequency across entire corpus
dtm = corpus.document_term_matrix(corpus.token_set)
tf = 1 + np.log(np.sum(dtm, axis=0))

# rank words - need to figure out which ones are which



################# Dictionary Methods - sentiment analysis #####################
#Load and format afin word dictionary
afinn_dict = convert_to_dictionary("./dictionaries/AFINN-111.txt")

# calculate sentiment of each article using the tf_idf matrix
sentiment_rank = corpus.weighted_dict_rank(afinn_dict,freq_rep = False )

# take out sentiment score and no of comments for plotting
sentiment = [float(x[3]) for x in sentiment_rank]
comments = [float(x[2]) for x in sentiment_rank]

# plot histogram
grdevices.png("./plots/afinn_hist.png")
rhist(robjects.FloatVector(sentiment), xlab="Sentiment", ylab="Number of documents", main="Distribution of Sentiment Scores")
grdevices.dev_off()

# plot afinn scores vs number of comments
grdevices.png("./plots/afinn_plot.png")
graphics.plot(sentiment, comments, xlab="Sentiment", ylab="Number of Comments", col="blue")
graphics.title("Sentiment vs Popularity of Waleed Aly Articles")
grdevices.dev_off()

################### Dictionary Methods - topic specific #######################
# load topic specific dictionaries
ethic_dict = read_dictionary('./dictionaries/ethics.csv')
politic_dict = read_dictionary('./dictionaries/politics.csv')

negative_dict = read_dictionary('./dictionaries/negative.csv')
positive_dict = read_dictionary('./dictionaries/positive.csv')
passive_dict = read_dictionary('./dictionaries/passive.csv')

# rank articles according to their term count for each particular dictionary
ethic_rank = corpus.dict_rank(ethic_dict, False)
politic_rank = corpus.dict_rank(politic_dict, False)
negative_rank = corpus.dict_rank(negative_dict, False)
positive_rank = corpus.dict_rank(positive_dict, False)
passive_rank = corpus.dict_rank(passive_dict, False)

# take out scores and no of comments for plotting
ethic_score = [float(x[3]) for x in ethic_rank]
ethic_comments = [float(x[2]) for x in ethic_rank]

politic_score = [float(x[3]) for x in politic_rank]
politic_comments = [float(x[2]) for x in politic_rank]

negative_score = [float(x[3]) for x in negative_rank]
negative_comments = [float(x[2]) for x in negative_rank]

positive_score = [float(x[3]) for x in positive_rank]
positive_comments = [float(x[2]) for x in positive_rank]

passive_score = [float(x[3]) for x in passive_rank]
passive_comments = [float(x[2]) for x in passive_rank]

# ethics histogram
grdevices.png("./plots/ethics_hist.png")
rhist(robjects.FloatVector(ethic_score), xlab="Ethics score", ylab="Number of documents", main="Distribution of Articles Relating to Ethics")
grdevices.dev_off()

# plot ethics score vs comments
grdevices.png("./plots/ethic_plot.png")
graphics.plot(ethic_score, ethic_comments, xlab="Ethics score", ylab="Number of Comments", col="blue")
graphics.title("Ethics vs Popularity of Waleed Aly Articles")
grdevices.dev_off()

# politics histogram
grdevices.png("./plots/politic_hist.png")
rhist(robjects.FloatVector(politic_score), xlab="Politics score", ylab="Number of documents", main="Distribution of Articles Relating to Politics")
grdevices.dev_off()

# plot politics score vs comments
grdevices.png("./plots/politic_plot.png")
graphics.plot(politic_score, politic_comments, xlab="Politics score", ylab="Number of Comments", col="blue")
graphics.title("Politics vs Popularity of Waleed Aly Articles")
grdevices.dev_off()

# histograms for positive, negative and passive 
grdevices.png("./plots/positive_hist.png")
rhist(robjects.FloatVector(positive_score), xlab="Positive score", ylab="Number of documents", main="Distribution of Articles Relating to Politics")
grdevices.dev_off()

grdevices.png("./plots/negative_hist.png")
rhist(robjects.FloatVector(negative_score), xlab="Negative score", ylab="Number of documents", main="Distribution of Articles Relating to Politics")
grdevices.dev_off()

grdevices.png("./plots/passive_hist.png")
rhist(robjects.FloatVector(passive_score), xlab="Passive score", ylab="Number of documents", main="Distribution of Articles Relating to Politics")
grdevices.dev_off()


# combine positive, negative and passive results into a dataframe
L = len(negative_score)
scores = negative_score + positive_score + passive_score
comments = negative_comments + positive_comments + passive_comments
dictionaries = ["negative"]*L + ["positive"]*L + ["passive"]*L
posnegpas = pd.DataFrame({'Score': scores, 'Comments': comments, 'Dictionary': dictionaries })

# create function to plot positive, negative and passive scores together using ggplot
plotFunc = robjects.r("""
 library(ggplot2)
 
function(df){
 p <- ggplot(df, aes(x=Score, y=Comments, col=factor(Dictionary))) +
 geom_point( ) +
 scale_colour_discrete(name="Dictionaries")
 
print(p)
 }
""")

# convert the testData to an R dataframe
robjects.pandas2ri.activate()
posnegpas_R = robjects.conversion.py2ri(posnegpas)
 
# run the plot function on the dataframe
grdevices.png('./plots/posnegpas_plot.png')
plotFunc(posnegpas_R)
grdevices.dev_off()



