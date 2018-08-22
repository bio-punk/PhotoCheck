import cv2 

def abs_slope(x1, y1, x2, y2):
	return (abs(y1 - y2)*1.0) / (abs(x1 - x2)*1.0)

def resize_width(image, width=1200):
	r = width * 1.0 / image.shape[1]
	dim = (width, int(image.shape[0] * r)) 
	resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
	return resized

def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x 
	h = rect.bottom() - y 
	return (x, y, w, h)

def get_center(arr_x, arr_y):
	x = 0
	y = 0
	for i in range(0, len(arr_x)):
		x = x + arr_x[i]
		y = y + arr_y[i]
	x = x / len(arr_x)
	y = y / len(arr_y)
	return [x, y]
