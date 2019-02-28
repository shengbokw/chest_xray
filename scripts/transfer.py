import os
import sys
from PIL import Image
from scipy import misc
import numpy as np
import pandas as pd
from subprocess import check_output
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt


# print(check_output(['ls', '../train'])).decode('utf8')

folder = '/Users/shengbo/shengbo/VU/ML/chest_xray/train/PNEUMONIA'
files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

image_width = 480
image_height = 320

train_pneumonia = np.ndarray(shape=(len(files), 1, image_height, image_width), dtype=np.float32)

min = 1000
max = -1

def distribution(gray_img, img_height, img_width):
    gray_distribution = [0] * 256
    for i in range(0, img_height):
        for j in range(0, img_width):
            gray_distribution[int(gray_img[i][j])] += 1
            # df[df['gray'] == int(gray_img[i][j])] += 1

    return gray_distribution

# gray_dist_pd = pd.DataFrame(gray_img)
data = {'gray': [i for i in range(0, 256)], 'count': gray_distribution}
df = pd.DataFrame(data)


df.plot()


for _file in files:
    img = load_img(folder + "/" + _file)
    # img.thumbnail((image_width, image_height))
    img_width = img.width
    img_height = img.height
    x = img_to_array(img)
    gray_img = x[:, :, 1]
    # train_pneumonia[i] = gray_img


file_normal = '/Users/shengbo/shengbo/VU/ML/chest_xray/train/NORMAL/NORMAL2-IM-0719-0001.jpeg'
img_normal = load_img(file_normal)
img_width_normal = img_normal.width
img_height_normal = img_normal.height
x_normal = img_to_array(img_normal)
gray_img_normal = x_normal[:, :, 1]

gray_distribution_normal = distribution(gray_img_normal, img_height_normal, img_width_normal)
data_normal = {'gray': [i for i in range(0, 256)], 'count': gray_distribution_normal}
df_normal = pd.DataFrame(data_normal)

df_normal.plot()
