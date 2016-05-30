# -*- coding: utf-8 -*-
"""
Created on Sun May 29 14:00:55 2016

@author: annekespeijers
"""

# import packages
from os import listdir
from os.path import join
import json
import io

# create list of file names
path = "./data/extracted"
files = [file for file in listdir(path)]

# parse all articles into one nexted list
articles = []

for file in files:
    print(file)
    file_path = join(path, file)
    fhandle = open(file_path, 'r') 
    
    # split into separate articles
    articles_str = fhandle.read().decode('utf-8').split('**********')
    #delete empty string 
    del articles_str[0]
    
    # create nested list for each article
    for article in range(len(articles_str)):
        
        # get the article
        original = articles_str[article]
    
        # take out the title
        title_split = original.split('********')
        title = title_split[0]
        remaining = title_split[1]
    
        # take out date
        date_split = remaining.split('******')
        date = date_split[0]
        remaining = date_split[1]

        # take out number of comments
        comments_split = remaining.split('****')
        comments = comments_split[0]
        
        # take article text and remove newline character
        text = comments_split[1].replace('\n', ' ')
        
        #paragraphs = re.compile('[\n]*').split(president_pattern[1])[1:-1]
        #del paragraphs[0]
        #del paragraphs[-1] - could also do this
                    
        # redefine the list of articles
        articles.append([ title, date, comments, text ])

    

# json.dumps() converts nested list to string
articlesJSON = json.dumps(articles, ensure_ascii=False)

# write json file
# nb. Would usually use json.dump() but there is a bug relating to utf8 encoding so use the io library.
with io.open('articles.txt', 'w', encoding="utf-8") as file:
    file.write(unicode(articlesJSON))


# load in file
with open('articles.txt') as file:    
    articles = json.load(file, encoding="utf-8")