from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import re
import urllib.request
from tqdm import tqdm
req = Request("https://shine.one-line.com/si_media/EU_SP/si_doc/")
html_page = urlopen(req)

soup = BeautifulSoup(html_page, "lxml")

links = []
for idx,link in tqdm(enumerate(soup.findAll('a'))):
    link = link.get('href')
    if 'pdf' in link.lower():
        # urllib.request.urlretrieve('https://shine.one-line.com/' + link, 
        #                      f"/home/tl/ShippingLabelGenerate/crawl_data/download_images/crawl_shine/{idx}.pdf")
        links.append(link)

print(len(links))
