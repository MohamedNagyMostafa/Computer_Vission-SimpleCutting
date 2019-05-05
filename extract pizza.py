import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('images/pizza_bluescreen.jpg')

print('type: ', type(image), ' size: ', image.shape)

# take a copy

image_copy = np.copy(image)
# Before
plt.imshow(image_copy)
plt.show()

#change BGR to RGB
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
#After
plt.imshow(image_copy)
plt.show()

#Threshold
lower_blue = np.array([0,0,220])
upper_blue = np.array([50,70,255])

#Mask: to isolate the area in range
mask = cv2.inRange(image_copy, lower_blue, upper_blue)

plt.imshow(mask, cmap='gray')
plt.show()

# Copy
masked_image = np.copy(image_copy)
masked_image[mask != 0] = [0,0,0]

plt.imshow(masked_image)
plt.show()

background_image = cv2.imread('images/background.jpg')
background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

background_image = background_image[0:mask.shape[0], 0:mask.shape[1]]

background_image[mask == 0] = [0,0,0]
background_image = background_image + masked_image

plt.imshow(background_image)
plt.show()