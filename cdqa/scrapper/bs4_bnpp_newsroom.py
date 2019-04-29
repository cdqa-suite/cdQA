import pandas as pd
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from lxml import html
import re
import os
import numpy as np

def fetch_bnpp_newsroom(output_dir=None):
    """
    [summary]
    
    Parameters
    ----------
    output_dir : [type], optional
        [description] (the default is None, which [default_description])
    
    Returns
    -------
    [type]
        [description]

    Examples
    --------
    >>> 

    """


    # get news homepage
    page = requests.get('https://group.bnpparibas/en/all-news')
    page_content = BeautifulSoup(page.text, 'html.parser')

    # get total number of pages
    pagination = str(page_content.find_all('ul', attrs={'class': 'real-pagination'}))
    nb_pages = max([int(i) for i in re.findall('(?<=data-to=")[0-9]*', pagination)])
    
    # get article links
    links = []

    for page_nb in tqdm(range(1, nb_pages+1)):

        page = requests.get('https://group.bnpparibas/en/all-news/{}'.format(str(page_nb)))
        page_content = BeautifulSoup(page.text, 'html.parser')

        articles = page_content.find_all('article')

        for article in articles:
            if article.find('a', attrs={'class': 'category'}):
                link = 'https://group.bnpparibas{}'.format(article.find_all('a')[1]['href'])
                links.append(link)

    # get articles content and metadata
    dates = []
    titles = []
    categories = []
    abstracts = []
    paragraphs = []

    for link in tqdm(links):

        page = requests.get(link)
        page_content = BeautifulSoup(page.text, 'html.parser')
        
        date = ' '.join(page_content.find('li', attrs={'class': 'date'}).text.split())
        title = page_content.h1.text
        category = ' '.join(page_content.find_all('a', attrs={'itemprop': 'item'})[1].text.split())
        try:
            abstract = page_content.find('div', attrs={'id': 'content'}).find('p', attrs={'class': 'abstract'}).text
            if not abstract:
                abstract = np.nan
        except:
            abstract = np.nan
        paragraph_raw = page_content.find('div', attrs={'id': 'content'}).find_all('p')
        paragraph_clean = [par.text.replace('\xa0', '') for par in paragraph_raw]

        dates.append(date)
        titles.append(title)
        categories.append(category)
        abstracts.append(abstract)
        paragraphs.append(paragraph_clean)

    # build final dataframe
    df = pd.DataFrame(columns=['date', 'title', 'category', 'link', 'abstract', 'paragraphs'])

    df['date'] = dates
    df['title'] = titles
    df['category'] = categories
    df['link'] = links
    df['abstract'] = abstracts
    df['paragraphs'] = paragraphs
    
    if output_dir:
        df.to_csv(os.path.join(output_dir, 'bnpp_newsroom-v1.0.csv'), index=False)

    return df
