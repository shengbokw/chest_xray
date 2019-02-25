import os
import sys
from PIL import Image
from scipy import misc
import numpy as np
import pandas as pd
from subprocess import check_output
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


# print(check_output(['ls', '../train'])).decode('utf8')

folder = '/Users/shengbo/shengbo/VU/ML/chest_xray/train/PNEUMONIA'
files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

image_width = 480
image_height = 320

train_pneumonia = np.ndarray(shape=(len(files), 1, image_height, image_width), dtype=np.float32)




def distribution(gray_img):
    gray_distribution = [0] * 255
    


for _file in files:
    img = load_img(folder + "/" + _file)
    # img.thumbnail((image_width, image_height))
    x = img_to_array(img)
    gray_img = x[:, :, 1]
    # train_pneumonia[i] = gray_img
