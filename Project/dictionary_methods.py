# import numpy as np
import codecs
import nltk
from nltk import PorterStemmer
import re
# import math
import pandas as pd
import os
import dill as pickle
from rpy2 import robjects
from rpy2.robjects.vectors import IntVector, FloatVector
from rpy2.robjects.packages import importr
import rpy2.robjects.pandas2ri
# from rpy2.robjects import Formula
# from rpy2.robjects.lib import grid


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

# load in corpus
with open('corpus.pkl', 'rb') as input:
    corpus = pickle.load(input)

###############################################################################
# The R 'print' function
rprint = robjects.globalenv.get("print")
rhist = robjects.r('hist')
stats = importr('stats')
grdevices = importr('grDevices')
base = importr('base')
ggplot2 = importr('ggplot2')
graphics = importr("graphics")

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



