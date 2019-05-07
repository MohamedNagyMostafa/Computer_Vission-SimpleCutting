import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('images/waffle.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Coners
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

image_copy = np.float32(image_copy)

dat = cv2.cornerHarris(image_copy, 5,5, 0.03)
dat = cv2.dilate(dat, None)

f, (ax1, ax2,ax3) = plt.subplots(1,3, figsize=(20,40))
ax1.imshow(image)
ax2.imshow(dat, cmap='gray')

#select brightness points

thresh = 0.01 * dat.max()

corner_image = np.copy(image)

for j in range(0, dat.shape[0]):
	for i in range(0, dat.shape[1]):
		if(dat[j,i] > thresh):
			cv2.circle(corner_image, (i,j), 1,(255,0,0),1)

ax3.imshow(corner_image)
plt.show()	