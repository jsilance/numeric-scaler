import datetime
from PIL import ImageGrab
import numpy as np
import cv2

# SCREEN_SIZE = (1920, 1080)

module_size = (5*128, 5*64) # output screen size 
capture_area = (0, 0, 1920, 1080)
offset = [1920, 0]


def img_input_calculator(img):
	img_in = np.array(img)
	img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
	cv2.imshow('Secret Capture', img_in)

def img_output_calculator(img):
	img_out = np.array(img)
	img_out = cv2.resize(img_out, module_size)
	img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
	img_out = cv2.copyMakeBorder(img_out, 0, 0, 12, 0, cv2.BORDER_CONSTANT, None, value = 0)
	cv2.imshow('Output', img_out)
	cv2.moveWindow('Output', offset[0]-20, offset[1]-32)

def controls():
	if cv2.waitKey(1) == 27:
		return 1

while True:
	img = ImageGrab.grab(bbox=capture_area)
	# img_input_calculator(img)
	img_output_calculator(img)

	if controls():
		break
	

cv2.destroyAllWindows()