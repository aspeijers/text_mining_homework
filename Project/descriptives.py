import os
import dill as pickle
import numpy as np
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import IntVector, FloatVector
import rpy2.robjects.pandas2ri
import pandas as pd

# set working directory
os.chdir("/Users/annekespeijers/Desktop/BGSE/Term3/TextMining/Homework/Project/")

# load in corpus
with open('./data/corpus.pkl', 'rb') as input:
    corpus = pickle.load(input)


# Number of words per article
dtm = corpus.document_term_matrix(corpus.token_set)
doc_word_counts = np.sum(dtm, axis=1)

# Zip's law
corpus_word_counts = pd.DataFrame({'Counts': -np.sort(-np.sum(dtm, axis=0)), 'Rank': range(corpus_word_counts.shape[0]) })

# import r devices
base = importr('base')
rbarplot = robjects.r('barplot')
rprint = robjects.globalenv.get("print")
rhist = robjects.r('hist')
graphics = importr("graphics")
ggplot2 = importr('ggplot2')


# plots
grdevices.png("./plots/article_lengths.png")
rhist(robjects.FloatVector(doc_word_counts), xlab="Article length (words)", ylab="Frequency", main="Distribution of article lengths")
grdevices.dev_off()


# create function to plot corpus word counts
plotFunc = robjects.r("""
 library(ggplot2)
 
function(df){
p <- ggplot(df, aes(x=Rank, y=Counts)) +
geom_line(colour="dodgerblue", size=1.5) +
scale_x_log10() +
scale_y_log10() +
labs(x="ranked words (log scale)",y="counts (log scale)")+
ggtitle("Corpus word counts") +
theme_bw()

 
print(p)
 }
""")

# convert the testData to an R dataframe
robjects.pandas2ri.activate()
corpus_word_counts_R = robjects.conversion.py2ri(corpus_word_counts)
 
# run the plot function on the dataframe
grdevices.png('./plots/word_counts.png')
plotFunc(corpus_word_counts_R)
grdevices.dev_off()
