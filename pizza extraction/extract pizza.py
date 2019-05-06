import cv2
import matplotlib.pyplot as plt
import numpy as np

# Reading
car_image = cv2.imread('images/car_green_screen.jpg')
background_image = cv2.imread('images/sky.jpg')
#converting
car_image = cv2.cvtColor(car_image,cv2.COLOR_BGR2RGB)
background_image = cv2.cvtColor(background_image,cv2.COLOR_BGR2RGB)


plt.imshow(background_image)
plt.show()

car_image_copy = np.copy(car_image)
plt.imshow(car_image_copy)
plt.show()
#Mask proparties
high_green = np.array([80, 255, 80])
lower_green = np.array([0, 170, 0])

mask = cv2.inRange(car_image_copy, lower_green, high_green)

plt.imshow(mask, cmap='gray')
plt.show()

car_image_copy[mask != 0] = [0,0,0]

plt.imshow(car_image_copy)
plt.show()

background_image = background_image[:car_image_copy.shape[0],:car_image_copy.shape[1]]

background_image[mask == 0] = [0,0,0]

background_image += car_image_copy

plt.imshow(background_image)
plt.show()