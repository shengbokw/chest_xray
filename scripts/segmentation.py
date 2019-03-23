import os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage import morphology, io, exposure, img_as_float, transform, img_as_ubyte
from matplotlib import pyplot as plt
import lung_size as ls


current_path = '/Users/shengbo/shengbo/VU/ML/chest_xray/lung-segmentation-2d/Demo/'
folder = '/Users/shengbo/shengbo/VU/ML/chest_xray/val/NORMAL'
files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
df = pd.DataFrame(data=files, columns={'img'})
df[df['img'] == '.DS_Store'] = None
df = df.dropna()
# csv_path = current_path + 'idx.csv'
# # Path to the folder with images. Images will be read from path + path_from_csv
# path = current_path + 'Data/'


def loadDataGeneral(df, path, im_shape):
    """
    reshaping images
    """
    X = []
    for i, item in df.iterrows():
        img = img_as_float(io.imread(path + '/' + item[0]))
        img = transform.resize(img, im_shape)
        img = exposure.equalize_hist(img)
        img = np.expand_dims(img, -1)
        X.append(img)
    X = np.array(X)
    X -= X.mean()
    X /= X.std()

    return X


def remove_small_regions(img, size):
    """
    Morphologically removes small (less than size)
    connected regions of 0s or 1s.
    """
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)

    return img


def distribution(gray_img):
    """
    calculate the distribution of img,
    total will be used on calculating fraction
    """
    img_shape = gray_img.shape
    gray_distribution = [0] * 256
    total = 0
    for i in range(0, img_shape[0]):
        for j in range(0, img_shape[1]):
            gray_distribution[int(gray_img[i][j])] += 1
            total += int(gray_img[i][j])

    return gray_distribution, total


def lung_density(pr, img):
    """calculate the density of two lungs."""
    density = 0
    size = 0
    img_shape = img.shape
    for i in range(0, img_shape[0]):
        for j in range(0, img_shape[1]):
            if pr[i][j] == 1:
                size += 1
                density += img[i][j]

    return density * 1.0 / (size + 1)


def extract_features(folder, df, savefile):
    """
    this function will combine features:
    gray scale value distribution,
    density of lungs,
    ...
    """
    features = []

    # Load test data
    im_shape = (256, 256)
    X = loadDataGeneral(df, folder, im_shape)

    # stop when it arrive the length of X
    n_test = X.shape[0]
    inp_shape = X[0].shape

    # Load model
    model_name = current_path + '../trained_model.hdf5'
    UNet = load_model(model_name)

    # For inference standard keras ImageGenerator can be used.
    test_gen = ImageDataGenerator(rescale=1.)

    i = 0
    for xx in test_gen.flow(X, batch_size=1):
        # feature = []
        img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0, 1))
        # I'm still thinking about how to deal with the gray scale
        # img = img_as_ubyte(img)
        img = img * 15
        img = img.astype(dtype=np.int8)
        pred = UNet.predict(xx)[..., 0].reshape(inp_shape[:2])
        pr = pred > 0.5
        pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))
        pr_int = np.array(pr, dtype=np.int8)

        dist, total = distribution(img)
        dist.append(total)
        dist.append(lung_density(pr_int, img))
        r, l, fraction = ls.size_of_lungs(pr_int)
        dist.append(r)
        dist.append(l)
        dist.append(fraction)

        features.append(dist)
        # np.savetxt('test.out', pr_int, delimiter='', fmt="%s")

        i += 1
        if i == n_test:
            break

    np_features = np.array(features)
    np.savetxt(savefile, np_features, delimiter=',', fmt="%s")
    # return features


extract_features(df, folder, 'normal_test.csv')
# plt.imshow(img, cmap='gray')
# df = pd.read_csv(csv_path)


"""visulization on test files"""
# plt.figure(figsize=(10, 10))
# for xx in test_gen.flow(X, batch_size=1):
#     img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0, 1))
#     pred = UNet.predict(xx)[..., 0].reshape(inp_shape[:2])
#
#     pr = pred > 0.5
#     ft = pred > 0.5
#     ft.fill(False)
#     pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))
#
#     prs.append(pr)
#
#     print(df.iloc[i][0])
#
#     if i < 4:
#         plt.subplot(4, 4, 4*i+1)
#         plt.title('Processed ' + df.iloc[i][0])
#         plt.axis('off')
#         plt.imshow(img, cmap='gray')
#
#         plt.subplot(4, 4, 4*i+3)
#         plt.title('Prediction')
#         plt.axis('off')
#         plt.imshow(pred, cmap='jet')
#
#         plt.subplot(4, 4, 4*i+4)
#         plt.title('Difference')
#         plt.axis('off')
#         plt.imshow(np.dstack((pr.astype(np.int8), ft.astype(np.int8), pr.astype(np.int8))))
#
#     i += 1
#     if i == n_test:
#         break
#
# plt.tight_layout()
# plt.savefig('results.png')
# plt.show()
#
# pr_int = np.array(pr, dtype=np.int8)
# np.savetxt('test.out', pr_int, delimiter='', fmt="%s")
