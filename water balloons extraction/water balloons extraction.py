import cv2
import matplotlib.pyplot as plt
import numpy as np

# Reading
image = cv2.imread('images/water_balloons.jpg')
#converting
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#copy
image_copy = np.copy(image)

plt.imshow(image_copy)
plt.show()

# Splitting channels

r = image_copy[:,:,0]
g = image_copy[:,:,1]
b = image_copy[:,:,2]

f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(20,10))

ax1.set_title('Red')
ax2.set_title('Green')
ax3.set_title('Blue')
ax4.set_title('Red_gray')
ax5.set_title('Green_gray')
ax6.set_title('Blue_gray')
ax1.imshow(r)
ax2.imshow(g)
ax3.imshow(b)
ax4.imshow(r, cmap='gray')
ax5.imshow(g, cmap='gray')
ax6.imshow(b, cmap='gray')

plt.show()

# HSV color space

image_copy_HSV = cv2.cvtColor(image_copy, cv2.COLOR_RGB2HSV)

h = image_copy_HSV[:,:,0]
s = image_copy_HSV[:,:,1]
v = image_copy_HSV[:,:,2]

f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(20,10))

ax1.set_title('Hue')
ax2.set_title('Saturation')
ax3.set_title('Value')
ax4.set_title('Hue_gray')
ax5.set_title('Saturation_gray')
ax6.set_title('Value_gray')
ax1.imshow(h)
ax2.imshow(s)
ax3.imshow(v)
ax4.imshow(h, cmap='gray')
ax5.imshow(s, cmap='gray')
ax6.imshow(v, cmap='gray')

plt.show()

# Cutting
lower_RGB = np.array([180, 0, 100])
upper_RGB = np.array([255, 255, 210])

mask_RGB = cv2.inRange(image_copy, lower_RGB, upper_RGB)

image_copy_RGB = np.copy(image_copy)

image_copy_RGB[mask_RGB == 0] = [0,0,0]

plt.imshow(image_copy_RGB)
plt.show()

lower_Hue = np.array([160,0,0])
high_Hue = np.array([180,255,255])

mask_HSV = cv2.inRange(image_copy_HSV, lower_Hue, high_Hue)

image_copy_HSV = np.copy(image_copy)

image_copy_HSV[mask_HSV == 0] = [0,0,0]

plt.imshow(image_copy_HSV)
plt.show()
