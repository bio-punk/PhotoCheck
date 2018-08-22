# -*- coding: utf-8 -*- 
import sys 
import numpy as np
import dlib
import cv2 
# from PIL import Image, ImageDraw, ImageFont 

FACE_POINTS         = list(range(17, 68))
MOUTH_POINTS        = list(range(48, 61))
RIGHT_BROW_POINTS   = list(range(17, 22))
LEFT_BROW_POINTS    = list(range(22, 27))
RIGHT_EYE_POINTS    = list(range(36, 42))
LEFT_EYE_POINTS     = list(range(42, 48))
NOSE_POINTS         = list(range(27, 35))
JAW_POINTS          = list(range(0, 17))

def abs_slope(x1, y1, x2, y2):
	return (abs(y1 - y2)*1.0) / (abs(x1 - x2)*1.0)

def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x 
	h = rect.bottom() - y 
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

def resize(image, width=1200):
	r = width * 1.0 / image.shape[1]
	dim = (width, int(image.shape[0] * r)) 
	resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
	return resized

def get_center(arr_x, arr_y):
	x = 0
	y = 0
	for i in range(0, len(arr_x)):
		x = x + arr_x[i]
		y = y + arr_y[i]
	x = x / len(arr_x)
	y = y / len(arr_y)
	return [x, y]

if len(sys.argv) < 3:
	print "Usage: %s <image file>" % sys.argv[0]
	sys.exit(1)

image_file = sys.argv[1]
out_file_name = sys.argv[2]
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

image = cv2.imread(image_file,cv2.IMREAD_COLOR)
image = resize(image, width=1200)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)

if len(rects) > 1:
	print("TooManyFaces")
	exit()
if len(rects) < 1:
	print("NoFaces")
	exit()

rect = rects[0]
shape = predictor(gray, rect)
shape = shape_to_np(shape)

(x, y, w, h) = rect_to_bb(rect)
cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
i = 0
left_eye_x = []
left_eye_y = []
right_eye_x = []
right_eye_y = []
mouth1_x = 9999999.99
mouth1_y = 9999999.99
mouth2_x = -9.9
mouth2_y = -9.9
for (x, y) in shape:
	cv2.rectangle(image, (x, y), (x+1, y+1), (127,255,127), 2)
	if i in [30]:
		nose_center_x = x
		nose_center_y = y

	if i in MOUTH_POINTS:
		if x < mouth1_x : mouth1_x = x
		if x > mouth2_x : mouth2_x = x
		if y < mouth1_y : mouth1_y = y
		if y > mouth2_y : mouth2_y = y

	if i in RIGHT_EYE_POINTS:
		right_eye_x.append(x)
		right_eye_y.append(y)
	if i in LEFT_EYE_POINTS:
		left_eye_x.append(x)
		left_eye_y.append(y)
	i = i + 1


[right_eye_center_x, right_eye_center_y] = get_center(right_eye_x, right_eye_y)
[left_eye_center_x, left_eye_center_y] = get_center(left_eye_x, left_eye_y)
eye_k = abs_slope(right_eye_center_x, right_eye_center_y, left_eye_center_x, left_eye_center_y)
print "2 eyes k = ", eye_k
right_eye_nose_k = abs_slope(right_eye_center_x, right_eye_center_y, nose_center_x, nose_center_y)
left_eye_nose_k = abs_slope(left_eye_center_x, left_eye_center_y, nose_center_x, nose_center_y)
print "2 eyes-nose k delta = ", abs(left_eye_nose_k - right_eye_nose_k)

cv2.line(image, (right_eye_center_x, right_eye_center_y), (left_eye_center_x, left_eye_center_y),(0,255,0),2)
cv2.line(image, (right_eye_center_x, right_eye_center_y), (nose_center_x, nose_center_y),(0,255,0),2)
cv2.line(image, (nose_center_x, nose_center_y), (left_eye_center_x, left_eye_center_y),(0,255,0),2)
cv2.rectangle(image, (mouth1_x, mouth1_y), (mouth2_x, mouth2_y), (0, 255, 0), 2)
cv2.line(image, (mouth1_x, mouth1_y), (mouth2_x, mouth2_y), (0,255,0),2)
cv2.line(image, (mouth1_x, mouth2_y), (mouth2_x, mouth1_y), (0,255,0),2)
cv2.line(image, (nose_center_x, nose_center_y), ((mouth1_x+mouth2_x)/2, (mouth1_y+mouth2_y)/2), (0,255,0),2)

cv2.imwrite(out_file_name, image)

cv2.imshow("img", image)
cv2.waitKey(0)
