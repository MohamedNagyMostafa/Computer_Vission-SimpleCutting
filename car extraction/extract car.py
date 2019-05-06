import cv2
import matplotlib.pyplot as plt
import numpy as np

# Reading
car_image = cv2.imread('images/car_green_screen.jpg')
background_image = cv2.imread('images/sky.jpg')
#converting
car_image = cv2.cvtColor(car_image,cv2.COLOR_BGR2RGB)
background_image = cv2.cvtColor(background_image,cv2.COLOR_BGR2RGB)
background_image = background_image[:car_image.shape[0], :car_image.shape[1]]

car_image_copy = np.copy(car_image)

#Mask proparties
high_green = np.array([80, 255, 80])
lower_green = np.array([0, 170, 0])

mask = cv2.inRange(car_image_copy, lower_green, high_green)

car_image_copy[mask != 0] = [0,0,0]

background_image_copy = np.copy(background_image)
background_image_copy = background_image_copy[:car_image_copy.shape[0],:car_image_copy.shape[1]]

background_image_copy[mask == 0] = [0,0,0]

background_image_copy += car_image_copy

f, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(20,10))
ax1.set_title('background')
ax2.set_title('car to extract')
ax3.set_title('car mask')
ax4.set_title('merging')
ax1.imshow(background_image)
ax2.imshow(car_image)
ax3.imshow(mask, cmap='gray')
ax4.imshow(background_image_copy)
plt.show()