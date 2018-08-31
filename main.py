# -*- coding: utf-8 -*- 
import sys 
import numpy as np
import dlib
import cv2 
from helper import *
import time
# from PIL import Image, ImageDraw, ImageFont 

FACE_POINTS         = list(range(17, 68))
MOUTH_POINTS        = list(range(48, 61))
RIGHT_BROW_POINTS   = list(range(17, 22))
LEFT_BROW_POINTS    = list(range(22, 27))
RIGHT_EYE_POINTS    = list(range(36, 42))
LEFT_EYE_POINTS     = list(range(42, 48))
NOSE_POINTS         = list(range(27, 35))
JAW_POINTS          = list(range(0, 17))


def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

def show_point(image, shape):    
	i = 0
	for (x, y) in shape:
		cv2.rectangle(image, (x, y), (x+1, y+1), (127,255,127), 2)
		cv2.putText(image, str(i), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 2)
		i = i + 1
	return image


timebegin = time.time()

if len(sys.argv) < 2:
	print "Usage: %s <image file>" % sys.argv[0]
	sys.exit(1)


image_file = sys.argv[1]
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

image = cv2.imread(image_file,cv2.IMREAD_COLOR)
image = resize_width(image, 800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)

if len(rects) > 1:
	print("TooManyFaces")
	exit()
if len(rects) < 1:
	print("NoFaces")
	exit()

timeend= time.time()

print int(1000*(timeend-timebegin)),'ms'

rect = rects[0]
shape = predictor(gray, rect)
shape = shape_to_np(shape)

image = show_point(image, shape)

cv2.imshow("img", image)
cv2.waitKey(0)
