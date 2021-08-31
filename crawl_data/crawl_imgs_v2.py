from core.images_download import images_download as imgs_download

def read_keyword(file_name):
    text_file = open(file_name, 'r')
    while True:
        line_data = text_file.readline()
        if not line_data:
            text_file.close()
            break
        yield line_data

if __name__ == '__main__':
    FILE_KEY_WORD = "carton.txt"
    NO_IMGS_PER_KEY = 3000

    key_words = read_keyword(FILE_KEY_WORD)
    response = imgs_download
    for kw in key_words:
        # response().download("backpack purse " + handbag_name, NO_IMGS_PER_BRAND,handbag_name)
        response().download(kw, NO_IMGS_PER_KEY,kw)