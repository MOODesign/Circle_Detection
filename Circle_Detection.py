import cv2
import numpy as np
import matplotlib.pyplot as plt

#importing our image
image = 'Circle_Detection/coins_1.jpg'
img = cv2.imread(image, 1)
#making a copy of the image.
img_orig = img.copy()
#converting the image to gray.
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#plotting the image.
plt.rcParams['figure.figsize'] = (16,9)
plt.imshow(img,cmap= 'gray')

#blurring the image to get accurate results and discard the details inside every coin.
img = cv2.GaussianBlur(img, (21,21), cv2.BORDER_DEFAULT)
plt.rcParams['figure.figsize'] = (16,9)
plt.imshow(img,cmap='gray')

#the houghCircles function to identify circles in an image.
all_circs = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 0.9,120,param1=50, param2=30, minRadius=60,maxRadius=120)
all_circs_rounded = np.uint16(np.around(all_circs))

#printing out the number of circles found.
print(all_circs_rounded)
print(all_circs_rounded.shape)
print('i have found ' + str(all_circs_rounded.shape[1]) + ' coins.')

#labeling every circle.
count = 1
for i in all_circs_rounded[0, :]:
    cv2.circle(img_orig, (i[0], i[1]), i[2], (50, 200 , 200), 5)
    cv2.circle(img_orig, (i[0], i[1]), 2, (255,0,0), 3)
    cv2.putText(img_orig, 'Coin ' + str(count), (i[0]-70, i[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,0,0), 2)
    count +=1

#plotting the images with the circles identified.
plt.rcParams['figure.figsize'] = (16,9)
plt.imshow(img_orig)
plt.show()
