from quantize import quantize, recreate_image
import matplotlib.pyplot as plt
import os
import numpy as np

n_colors = 2

def extract_graph(img, background):
    graph = []
    for j in range(img.shape[1]):
        bar = img[:, j]
        current_color = bar[0]
        y_start, y_end = 0, 0
        for i in range(img.shape[0]):
            if not bar[i] == current_color:
                if current_color != background:
                    graph.append([j, (y_end+y_start)/2])
                current_color = bar[i]
                y_start = i
            y_end += 1
    return np.array(graph)


img = plt.imread('demo_image.png')
codebook_random, kmeans, labels, labels_random, w, h = quantize(img, n_colors)
newimg = recreate_image(kmeans.cluster_centers_, labels, w, h)
graph = extract_graph(newimg, 0)
# Display all results, alongside original image
plt.subplot(1, 2, 1)
plt.axis('off')
plt.title('Original image (a lot of colors)')
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.axis('off')
plt.title(f'Quantized image ({n_colors} colors, K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
plt.scatter(graph[:, 0], graph[:, 1])
plt.show()
