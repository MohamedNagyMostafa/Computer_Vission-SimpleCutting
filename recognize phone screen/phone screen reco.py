import cv2
import numpy as np
import matplotlib.pyplot as plt

# Importing
phone_image = cv2.imread('images/phone.jpg')
phone_image = cv2.cvtColor(phone_image, cv2.COLOR_BGR2RGB)
phone_image_cp = np.copy(phone_image)

plt.imshow(phone_image_cp)
plt.show()


#Edge detection
low_threshold = 80
high_threshold = 140

#Gray
phone_image_cp = cv2.cvtColor(phone_image_cp, cv2.COLOR_RGB2GRAY)

phone_image_edges = cv2.Canny(phone_image_cp, low_threshold, high_threshold)

plt.imshow(phone_image_edges, cmap='gray')
plt.show()

#Hough transform
#Resolution
rho = 1
theta = np.pi/180
threshold = 60
min_line_length = 100
max_line_gap = 5 

lines = cv2.HoughLinesP(phone_image_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

line_image = np.copy(phone_image)

for line in lines:
	for x1, y1, x2, y2 in line:
		cv2.line(line_image, (x1,y1),(x2,y2), (255,0,0), 5)

plt.imshow(line_image)
plt.show()