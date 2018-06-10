import cv2
import numpy as np
from matplotlib import pyplot as plt

wiki = cv2.imread('wiki.jpg', cv2.IMREAD_GRAYSCALE)
hist, bins = np.histogram(wiki.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_norm = cdf * 255 / cdf[-1]
wiki_hist_eq = cdf_norm[wiki]

fig, axes = plt.subplots(2, 2)

axes[0][0].set_xticks([])
axes[0][0].set_yticks([])
axes[0][0].set_ylabel('(a) Original Image')
axes[0][0].imshow(wiki, cmap='gray')

axes[0][1].hist(wiki.ravel(), 256, [0, 256])

axes[1][0].set_xticks([])
axes[1][0].set_yticks([])
axes[1][0].set_ylabel('(b) Histogram Equalized')
axes[1][0].imshow(wiki_hist_eq, cmap='gray')

axes[1][1].hist(wiki_hist_eq.ravel(), 256, [0, 256])

fig.savefig('histograms.png', bbox_inches='tight')
plt.show()
