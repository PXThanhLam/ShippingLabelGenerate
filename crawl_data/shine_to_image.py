import os
from pdf2image import convert_from_path
from tqdm import tqdm
root = '/home/tl/ShippingLabelGenerate/crawl_data/download_images/crawl_shine/'
save = '/home/tl/ShippingLabelGenerate/crawl_data/download_images/crawl_shine_image/'
idx = 0
for pdf in tqdm(os.listdir(root)):
    images = convert_from_path(root + pdf )
    for i in range(len(images)):
        images[i].save(save + 'page_'+ str(idx) +'.jpg', 'JPEG')
        idx += 1
    if idx == 10:
        break
    