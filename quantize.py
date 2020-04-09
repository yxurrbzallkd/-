import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle


def quantize(img, n_colors=64):
    # Load the Summer Palace photo
    img = np.array(img, dtype=np.float64) / 255

    # Load Image and transform to a 2D numpy array.
    w, h, d = tuple(img.shape)
    assert(d == 3)
    image_array = np.reshape(img, (w * h, d))
    print("Fitting model on a small sub-sample of the data")
    image_array_sample = shuffle(image_array)[:min(2000, w*h)]
    kmeans = KMeans(n_clusters=n_colors).fit(image_array_sample)

    print("Predicting color indices on the full image (k-means)")
    labels = kmeans.predict(image_array)

    codebook_random = shuffle(image_array, random_state=0)[:n_colors]
    print("Predicting color indices on the full image (random)")
    labels_random = pairwise_distances_argmin(codebook_random,
                                            image_array,
                                            axis=0)
    return codebook_random, kmeans, labels, labels_random, w, h


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    codebook = [i for i in range(len(codebook))]
    image = np.zeros((w, h))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    print('recreated image')
    return image

def recreate_to_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    codebook = [i for i in range(len(codebook))]
    image = np.zeros((w, h))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    print('recreated image')
    return image
