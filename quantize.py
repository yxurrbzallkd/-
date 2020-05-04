import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from time import time
from skimage.color import rgb2lab, lab2rgb, rgb2hsv

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


def recreate_2_image(labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    image = np.zeros((w, h))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = labels[label_idx]
            label_idx += 1
    return image

def normalize(img):
    sample = shuffle(np.reshape(img, (img.shape[0]*img.shape[1], 3)))[:min(500,img.shape[0]*img.shape[1])]
    img[:, :, 0] = (img[:, :, 0] - np.mean(sample[:, 0]))
    img[:, :, 1] = (img[:, :, 1] - np.mean(sample[:, 1]))
    img[:, :, 2] = (img[:, :, 2] - np.mean(sample[:, 2]))
    return img

def quantize_a_piece(window):
    width, height, depth = window.shape
    normalize(window)
    image_array = np.reshape(window, (width * height, depth))
    image_array[np.isnan(image_array)] = 0
    image_array_sample = shuffle(image_array)[:min(max(400, width * height), 1000)]
    kmeans = KMeans(n_clusters=2).fit(image_array_sample)
    labels = kmeans.predict(image_array)

    if np.sum(labels) > len(labels)//2:
        if len(labels)-np.sum(labels) < len(labels)//6: 
            labels = np.where(labels==0, 3, labels)
            labels = np.where(labels==1, 0, labels)
            labels = np.where(labels==3, 1, labels)
    return recreate_2_image(labels, width, height)

def quantize(img, h=4, w=4):
    img = np.array(img, dtype=np.float64)

    h, w = img.shape[0]//h, img.shape[1]//w
    newimg = np.zeros(shape=img.shape)

    for i in range(0, img.shape[0], h):
        for j in range(0, img.shape[1], w):
            window = img[i:i+h, j:j+w]
            newimg[i:i+h, j:j+w] = normalize(window)

    return quantize_a_piece(newimg)

'''
from images import images
for i in images[4:]:
    img = plt.imread(i)
    s = quantize(img)
    #img = rgb2hsv(img)
    #img = rgb2lab(img)
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.title('Original image (a lot of colors)')
    plt.imshow(img)

    # Display all results, alongside original image    
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.title(f'Quantized image ({2} colors, K-Means)')
    plt.imshow(s)
    print('showing')
    plt.show()
'''