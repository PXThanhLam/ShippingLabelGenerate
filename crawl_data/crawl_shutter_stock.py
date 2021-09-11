from os import path
import requests
from bs4 import BeautifulSoup
import urllib.request
from tqdm import tqdm
import numpy as np
import os

page = 1
count = 0
while True:
    get_soup = requests.get(
        f"https://www.shutterstock.com/search?searchterm=cardboard&sort=popular&image_type=all&search_source=base_landing_page&language=en&page={page}"
        ,headers={"User-Agent": "XY"})
    scraper = BeautifulSoup(get_soup.text, "html.parser")
    img_container = scraper.find_all("img", {"class":"z_h_9d80b z_h_2f2f0"})
    for j in tqdm(range(0, len(img_container)-1)):
        img_src = img_container[j].get("src")
        name = img_src.rsplit("/", 1)[-1] 

        base = name.split('.')[0].split('-')[-1]
        if np.random.rand() >=0.3:
            img_src = f'https://image.shutterstock.com/shutterstock/photos/{base}/display_1500/' + name
        else:
            img_src = f'https://image.shutterstock.com/image-photo/image-photo' + name

        try:
            urllib.request.urlretrieve(img_src, f'/content/Backround/shuyyer_{os.path.basename(img_src)}.jpg')
            count += 1
        except Exception as e:
            print(e)
    page +=1
