# This script is to scrape articles that were rejected in the first round of scraping due to 
# an error whilst running the 'format_content' function. 
# An if statement has been added so that if the number of comments cannot be scraped, the 
# article is assigned a comment count of 0. 

#import modules 
import urllib2
from bs4 import BeautifulSoup as bs
import os
import numpy as np
import sys
import time, random
import re



def get_html(url):
    """
    This function gets the whole html content of a specific url
    """
    response = urllib2.urlopen(url) 
    html = response.read()
    return html 



def format_content(html):
    """
    This gets the title, date, number of shares on facebook and text from an article.
    """
    soup = bs(html,"lxml")
        
    # find article title
    title = soup.find('h1', {'class': 'cN-headingPage'}).get_text().encode('utf-8')
    
    # find date
    date_str = soup.find_all('time', {'itemprop': 'datePublished'})
    date = date_str[0].get('datetime')
    
    # find number of comments (if I can't get fb shares)
    try:
        # try to scrape no of comments
        comments_str = soup.find('li', {'class': 'comments'}).get_text()
        comments = re.sub('[^0-9]', " ", comments_str).split()
        comments_no = comments[0].encode('utf-8')
    
    except:
        comments_no = "0"
        
    # find content of the article
    article_body = soup.find('div', {"class": "articleBody"}) 
    paragraphs = article_body.find_all('p', recursive=False)
    article_text = " "
    for i in range(len(paragraphs)-1): # don't include final byline 
        paragraph = paragraphs[i].get_text().encode('utf-8')
        article_text = article_text + paragraph
    
    # insert 10 stars for title, 8 for date, 6 for comments, 4 for text
    return ('**********' + title + '********' + date + '******' + comments_no + '****' + article_text)
 

 
def scrape_article_info(urls, file_prefix):
    '''
    This function takes a list of article urls and scrapes the html content of
    each one. Logs are kept of completed and rejected urls. 
    '''
    articles = []
    
    for idx, url in enumerate(urls):
        #random delay between 1 and 2 secs
        delay = random.randint(1,2)
        #pause execution for delay seconds
        time.sleep(delay)
        print 'file: ' + file_prefix + str(idx) + ' delay= ' + str(delay) 
        try:
            #try to retrieve information from url
            article_html = get_html(url)
            
            #add completed url to the log of completed urls
            # open a portal             
            with open("./completed_html_article_urls_2.txt", "a") as complete_file:
                complete_file.write(url + '\n')
                #close the portal
                complete_file.close()
                
            try:
                # try to format the content
                articles.append(format_content(article_html))
                
                # add articles that are able to be formatted to a log
                with open("./completed_formatting_article_urls_2.txt", "a") as complete_file:
                    complete_file.write(url + '\n')
                    complete_file.close()
            
            except:
                # add articles that are unable to be formatted to a log
                with open("./rejected_formatting_article_urls_2.txt", "a") as rejected_file:
                    rejected_file.write(url + '\n')
                    rejected_file.close()
                    
        except:
            #add rejected urls to the log of rejected urls
            with open("./rejected_html_article_urls_2.txt", "a") as rejected_file:
                rejected_file.write(url + '\n')               
                rejected_file.close()
        
        if idx % 10 == 0 and idx != 0: 
            
            #periodically write the data to file 
            file_name = file_prefix + str(idx) + '.gz'
            np.savetxt(file_name, articles, delimiter='\n', fmt='%s')
            #reinitialise list          
            articles = []
            
            #add the index of last file to be written to disk
            with open("./saved_data_index_2.txt", "a") as saved_file:
                saved_file.write(file_prefix + str(idx) + '\n')
                saved_file.close()
    
    #Save remaining data to file    
    file_name = file_prefix + str(idx) + '.gz'
    np.savetxt(file_name, articles, delimiter='\n', fmt='%s')
    with open("./saved_data_index_2.txt", "a") as saved_file:
                saved_file.write(file_prefix + str(idx) + '\n')
                saved_file.close()
        
    
    
##########################################################################################
# read in the rejected urls 
urls_str = open('./rejected_formatting_article_urls.txt', 'r').read()
# convert to list
urls = urls_str.split('\n')
del urls[-1]

scrape_article_info(urls, "waleed_articles_1_")