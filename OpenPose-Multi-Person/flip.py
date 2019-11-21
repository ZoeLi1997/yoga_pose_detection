from PIL import Image
from PIL import ImageOps
import os

root_path = '../datasets/approved/'
flip_path = '../datasets/fliped_images/'
paths = ['mountain', 'tree', 'triangle', 'warrior2', 'natarajasana', 'side_plank']

for i in range(len(paths)):
    path = root_path + paths[i]
    for root,d_names,f_names in os.walk(path):
        #traverse each image
        for image in f_names:
            print("Processing image: {}".format(image))
            image_path = root_path + paths[i] + "/" + image
            im = Image.open(image_path)
            im = ImageOps.mirror(im)
            flipped_path = flip_path + paths[i] + "/" + "flipped_" + image
            im.save(flipped_path)