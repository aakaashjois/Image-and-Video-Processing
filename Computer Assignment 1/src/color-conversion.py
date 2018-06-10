import cv2
import numpy as np
from matplotlib import pyplot as plt


colorsBGR = cv2.imread('colors.jpg', cv2.IMREAD_COLOR)
colorsHSV = cv2.cvtColor(colorsBGR, cv2.COLOR_BGR2HSV)

blueRange = np.array([[110, 50, 50], [130, 255, 255]])
blueMask = cv2.inRange(colorsHSV, blueRange[0], blueRange[1])
colorsBlueMask = cv2.bitwise_and(colorsBGR, colorsBGR, mask=blueMask)
cv2.imwrite('blueMask.jpg', blueMask)
cv2.imwrite('colorsBlueMask.jpg', colorsBlueMask)

ax1 = plt.subplot(1, 3, 1)
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
ax1.set_title('(a) Original Image')
ax1.imshow(colorsBGR)

ax2 = plt.subplot(1, 3, 2)
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax2.set_title('(b) Mask Image')
ax2.imshow(blueMask, cmap='gray')

ax3 = plt.subplot(1, 3, 3)
ax3.get_xaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)
ax3.set_title('(c) Segmented Image')
ax3.imshow(cv2.cvtColor(colorsBlueMask, cv2.COLOR_BGR2RGB))

plt.show()

'''
To find the threshold value of any color,
1. Convert color to HSV format
2. Lower Threshold = [H-10, 100, 100]
3. Upper Threshold = [H+10, 100, 100]

Source: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#how-to-find-hsv-values-to-track
'''
