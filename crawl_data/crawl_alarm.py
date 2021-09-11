import requests
from bs4 import BeautifulSoup
import urllib.request
from tqdm import tqdm
import numpy as np
import bs4
print(bs4.__version__)

phrase = 'carton-boxes'
page = 1
count = 0
while True:
    get_soup = requests.get(
        f"https://www.alamy.com/search.html?&qt=cardboard&qt_raw=cardboard&pn={page}")
    soup = BeautifulSoup(get_soup.text, "html.parser")
    pics = soup.findAll("img",{'class' : 'lazyload'})
    for pic in tqdm(pics):
        try:
            src = pic.get("data-src")
            alt = pic.get("alt").split('-')[0].strip().lower().replace(' ','-')
            src = src.replace('zooms/6','zooms/3')
            fold = src.split('/')[-1].split('.')[0].upper()
            src = f'https://c7.alamy.com/comp/{fold}/{alt}-{fold}.jpg'
            urllib.request.urlretrieve(src, 
                             f"/content/Backround/crawl_alam_{count}.jpg")
            count += 1
        except:
            pass
    page +=1
