# python3

class img_buffer():
    imgs = None
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_key = img_key

    def getimage(self, img_key):
        return self.imgs[img_key]

