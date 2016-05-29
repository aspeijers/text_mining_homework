#import modules 
import urllib2
from bs4 import BeautifulSoup as bs
import os
import numpy as np
import sys
import time, random
import re


def get_no_pages(first_url):
    """
    This function takes the first results page of Waleed Aly's articles and
    outputs the total number of result pages.
    """
    response = urllib2.urlopen(first_url) 
    html = response.read()
    soup = bs(html,"lxml")
    pages = []
    for page in soup.find_all('li',{"class": "page"}):
        pages.append(page.get_text())
    no_pages = int(max(pages))
    return(no_pages)
    


def create_result_page_urls(first_url, no_pages):
    """
    This function creates artificially a list of urls for 
    the different result pages. It takes as an argument the url of the first 
    results page and the number of results pages. 
    """
    page_urls = [first_url]
    for i in range((no_pages-1)):
        url = first_url+"?offset="+str((i+1)*20)
        page_urls.append(url)
    return(page_urls)
    
    

def get_article_urls(results_page_url,website_prefix):
    """
    This function goes to one of the result pages and takes the urls of all the
    articles listed on that particular page. The output is a list with those 
    urls.
    Note. For my project, webiste_prefix = ""
    """
    response = urllib2.urlopen(results_page_url) #successful opening
    html = response.read()   
    soup = bs(html,"lxml")
    lead_link = soup.find_all('div', {"class": "wof"})
    links = soup.find_all('div',{"class": "cN-storyHeadlineLead cfix"})   
    links.extend(lead_link)
    urls = []
    for link in links:
        href= link.find('a').get('href')
        href = website_prefix + href 
        urls.append(href)
    return urls



def scrape_article_urls(urls):
    '''
    This function takes a list of result page urls and returns a list of the 
    article urls scraped from each one. Sucessfully scraped results page urls 
    are saved in 'completed_result_page_urls.txt'. Rejected result page urls 
    are saved in 'rejected_result_page_urls.txt'.
    '''
    article_urls = []
    
    #enumerate creates tuples     
    for idx, url in enumerate(urls):
        
        #random delay between 1 and 2 secs
        delay = random.randint(1,2)
        
        #pause execution for delay seconds
        time.sleep(delay)
        print 'file: ' + str(idx) + ' delay= ' + str(delay) 
        try:
            # extract the article urls and add to the list
            article_urls = article_urls + get_article_urls(url,"")
            
            #add completed url to the log of completed urls
            # open a portal             
            with open("./completed_result_page_urls.txt", "a") as complete_file:
                complete_file.write(url + '\n')
                
                #close the portal
                complete_file.close()
        except:
            #add rejected urls to the log of rejected urls
            # open a portal             
            with open("./rejected_result_page_urls.txt", "a") as rejected_file:
                rejected_file.write(url + '\n')
                
                #close the portal                
                rejected_file.close()
        
    return(article_urls)
        

     
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
    comments_str = soup.find('li', {'class': 'comments'}).get_text()
    comments = re.sub('[^0-9]', " ", comments_str).split()
    comments_no = comments[0].encode('utf-8')
        
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
            with open("./completed_html_article_urls.txt", "a") as complete_file:
                complete_file.write(url + '\n')
                #close the portal
                complete_file.close()
                
            try:
                # try to format the content
                articles.append(format_content(article_html))
                
                # add articles that are able to be formatted to a log
                with open("./completed_formatting_article_urls.txt", "a") as complete_file:
                    complete_file.write(url + '\n')
                    complete_file.close()
            
            except:
                # add articles that are unable to be formatted to a log
                with open("./rejected_formatting_article_urls.txt", "a") as rejected_file:
                    rejected_file.write(url + '\n')
                    rejected_file.close()
                    
        except:
            #add rejected urls to the log of rejected urls
            with open("./rejected_html_article_urls.txt", "a") as rejected_file:
                rejected_file.write(url + '\n')               
                rejected_file.close()
        
        if idx % 10 == 0 and idx != 0: 
            
            #periodically write the data to file 
            file_name = file_prefix + str(idx) + '.gz'
            np.savetxt(file_name, articles, delimiter='\n', fmt='%s')
            #reinitialise list          
            articles = []
            
            #add the index of last file to be written to disk
            with open("./saved_data_index.txt", "a") as saved_file:
                saved_file.write(file_prefix + str(idx) + '\n')
                saved_file.close()
    
    #Save remaining data to file    
    file_name = file_prefix + str(idx) + '.gz'
    np.savetxt(file_name, articles, delimiter='\n', fmt='%s')
    with open("./saved_data_index.txt", "a") as saved_file:
                saved_file.write(file_prefix + str(idx) + '\n')
                saved_file.close()
        
    
    
def main(first_url):
    """
    Wrapper function for running from the command line. 
    """
    # find number of results pages
    no_pages = get_no_pages(first_url)
    
    # construct list of all result pages urls
    result_page_urls = create_result_page_urls(first_url,no_pages)
    
    # get a list of all article urls
    article_urls = scrape_article_urls(result_page_urls)
    
    # check if there are rejected result page urls and STOP if there are
    if os.path.isfile("./rejected_result_page_urls.txt"):
        print("Error in extracting page urls")        
        return(article_urls)
    scrape_article_info(article_urls, "waleed_articles_")

#automate the function    

if __name__=="__main__":
    main(sys.argv[1])
    
        
       
    
    
#### Trial

#first_url = "http://www.theage.com.au/comment/by/waleed-aly"
#article_url = "http://www.theage.com.au/comment/malcolm-turnbull-stop-dithering-on-tax-reform-and-tell-us-what-you-really-think-20160217-gmx1qs"