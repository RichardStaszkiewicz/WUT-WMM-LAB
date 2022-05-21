# -*- coding: utf-8 -*-
"""
# Analiza obrazu - detekcja twarzy
"""

import cv2
#from google.colab.patches import cv2_imshow
import dlib

"""## Obraz testowy"""

img = cv2.imread("2_Demonstration_Demonstration_Or_Protest_2_1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imshow("img", img)

"""
## Kaskada Haara

Opracowano na podstawie: https://www.pyimagesearch.com/2021/04/05/opencv-face-detection-with-haar-cascades/
"""

haar_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

rects = haar_detector.detectMultiScale(gray, scaleFactor=1.05,
	                                minNeighbors=5, minSize=(30, 30),
	                                flags=cv2.CASCADE_SCALE_IMAGE)

img_haar = img.copy()
for (x, y, w, h) in rects:
  cv2.rectangle(img_haar, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("img_haar", img_haar)

"""
## Histogram zorientowanych gradientów (HOG) z maszyną wektorów nośnych (SVM)

Opracowano na podstawie: https://www.pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/
"""

hog_svm_detector = dlib.get_frontal_face_detector()

hog_svm_rects = hog_svm_detector(rgb, 1)

def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)

hog_svm_boxes = [convert_and_trim_bb(img, r) for r in hog_svm_rects]

img_hog_svm = img.copy()
for (x, y, w, h) in hog_svm_boxes:
	cv2.rectangle(img_hog_svm, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
cv2.imshow("img_hog_svm", img_hog_svm)

"""
## Splotowa sieć neuronowa (CNN)

Opracowano na podstawie: https://www.pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/
"""

cnn_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

cnn_rects = cnn_detector(rgb, 1)

cnn_boxes = [convert_and_trim_bb(img, r.rect) for r in cnn_rects]

img_cnn = img.copy()
for (x, y, w, h) in cnn_boxes:
	cv2.rectangle(img_cnn, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
cv2.imshow("img_cnn", img_cnn)

cv2.waitKey(0)
cv2.destroyAllWindows()