import time,re,os
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup
import base64
import uuid

def decode_image(img_dir,src):
    """
         Decoded picture
         : Param src: Image coding
        eg:
            src="data:image/gif;base64,R0lGODlhMwAxAIAAAAAAAP///
                yH5BAAAAAAALAAAAAAzADEAAAK8jI+pBr0PowytzotTtbm/DTqQ6C3hGX
                ElcraA9jIr66ozVpM3nseUvYP1UEHF0FUUHkNJxhLZfEJNvol06tzwrgd
                LbXsFZYmSMPnHLB+zNJFbq15+SOf50+6rG7lKOjwV1ibGdhHYRVYVJ9Wn
                k2HWtLdIWMSH9lfyODZoZTb4xdnpxQSEF9oyOWIqp6gaI9pI1Qo7BijbF
                ZkoaAtEeiiLeKn72xM7vMZofJy8zJys2UxsCT3kO229LH1tXAAAOw=="

         : Return: str saved to a local file name
    """
    # 1, information extraction
    result = re.search("data:image/(?P<ext>.*?);base64,(?P<data>.*)", src, re.DOTALL)
    if result:
        ext = result.groupdict().get("ext")
        data = result.groupdict().get("data")

    else:
        raise Exception("Do not parse!")

    # 2, base64 decoding
    img = base64.urlsafe_b64decode(data)

    # 3, the binary file is saved
    filename = "{}.{}".format(uuid.uuid4(), ext)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_file = os.path.join(img_dir,filename)
    with open(img_file, "wb") as f:
        f.write(img)

    return filename


def encode_image(filename):
    """
         Coded picture
         : Param filename: str local image file name
         : Return: the string str encoded
        eg:
        src="data:image/gif;base64,R0lGODlhMwAxAIAAAAAAAP///
            yH5BAAAAAAALAAAAAAzADEAAAK8jI+pBr0PowytzotTtbm/DTqQ6C3hGX
            ElcraA9jIr66ozVpM3nseUvYP1UEHF0FUUHkNJxhLZfEJNvol06tzwrgd
            LbXsFZYmSMPnHLB+zNJFbq15+SOf50+6rG7lKOjwV1ibGdhHYRVYVJ9Wn
            k2HWtLdIWMSH9lfyODZoZTb4xdnpxQSEF9oyOWIqp6gaI9pI1Qo7BijbF
            ZkoaAtEeiiLeKn72xM7vMZofJy8zJys2UxsCT3kO229LH1tXAAAOw=="

    """
    # 1, file read
    ext = filename.split(".")[-1]

    with open(filename, "rb") as f:
        img = f.read()

    # 2, base64 encoded
    data = base64.b64encode(img).decode()

    # 3, the picture encoding string concatenation
    src = "data:image/{ext};base64,{data}".format(ext=ext, data=data)
    return src

if __name__ == '__main__':
    handbagnames = ["Celine","Chanel","Givenchy",
                    "Gucci","Hermes","Brighton",
                    "Burberry","Calvin Klein",
                    "Chloé","Coach","Coach Factory",
                    "Cole Haan","Dooney & Bourke",
                    "Fendi","Fossil","Furla",
                    "Kate Spade New York","Longchamp",
                    "Louis Vuitton","Marc by Marc Jacobs",
                    "MICHAEL Michael Kors","Nine West",
                    "Prada","Rebecca Minkoff",
                    "Salvatore Ferragamo","Ted Baker",
                    "Tory Burch","Vera Bradley"]
    urls=[]

    for handbag in handbagnames:
        urls += ["https://www.google.com/search?q=handbag%20purse%20"+handbag+"&source=lnms&tbm=isch"]
    driver = webdriver.Chrome()
    for (handbagname,url) in zip(handbagnames,urls):
        print("url: ",url)
        print("handbagname: ",handbagname)
        img_dir = os.path.join("images",handbagname)
        driver.get(url)
        WebDriverWait(driver, timeout=200)
        time.sleep(5)
        page_source = driver.page_source

        soup = BeautifulSoup(page_source, "html.parser")
        images = [a.get('src') for a in soup.findAll('img', src=re.compile(r'base64'))]
        skipped_imgs=0
        item = 0
        for img in images:
            imgdata = decode_image(img_dir,img)
            item +=1
