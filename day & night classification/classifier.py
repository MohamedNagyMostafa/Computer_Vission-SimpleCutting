import cv2 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

def load_data(dir):
	im_list= []
	labels = ['day', 'night']

	for label in labels:
		for file in glob.glob(os.path.join(dir, label, "*")):

			image = mpimg.imread(file)

			if not image is None:
				im_list.append((image, label))

	return im_list

def preprocessing(images):
	stand_list = []

	for image_data in images:
		image = image_data[0]
		label = image_data[1]

		label_n = encoding(label)
		image_n = resizing(image)

		stand_list.append((image_n, label_n))

	return stand_list

def resizing(image):
	cv2.resize(image, (1100,600))

	return image

def encoding(label):
	return (label == 'day')

def avg_brightness(rgb_image):
    
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    sum_brightness = np.sum(hsv[:,:,2])
    
  
    avg = sum_brightness/(len(rgb_image)*len(rgb_image[0]))
    
    return avg

def estimated_image(image, threshold):
    avg = avg_brightness(image)
    
    return avg > threshold

def accuracy(number_correct_images, total_images):
    print('accuracy : ', (number_correct_images/ total_images) * 100, '%')

def train(images, threshold):
    correct_class = 0
    for image_data in images:
        image = image_data[0]
        label = image_data[1]
        
        extimated_label = estimated_image(image, threshold)
        
        correct_class += (extimated_label == label)
    return correct_class

def getThreshold(images):
	max_night_avg = 0
	min_day_avg = 255

	for image in images:
		avg = avg_brightness(image[0])
		if image[1] == 1 and min_day_avg > avg:
			min_day_avg = avg
		elif image[1] == 0 and max_night_avg < avg:
			max_night_avg = avg

	return np.mean([max_night_avg, min_day_avg])

image_dir_training = "images/training/"
image_dir_test = "images/test/"

IMAGE_LIST = load_data(image_dir_training)
IMAGE_LIST_TEST = load_data(image_dir_test)

STANDARDIZED_LIST = preprocessing(IMAGE_LIST)
STANDARDIZED_LIST_TEST = preprocessing(IMAGE_LIST_TEST)

image_num = 0
test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]

hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('Standardized image')
ax1.imshow(test_im)
ax2.set_title('H channel')
ax2.imshow(h, cmap='gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')


threshold = getThreshold(STANDARDIZED_LIST)
correct = train(STANDARDIZED_LIST, threshold)
accuracy(correct, len(STANDARDIZED_LIST))

correct = train(STANDARDIZED_LIST_TEST, threshold)
accuracy(correct, len(STANDARDIZED_LIST_TEST))