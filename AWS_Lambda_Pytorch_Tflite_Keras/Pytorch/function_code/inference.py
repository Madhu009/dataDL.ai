
import cv2, torch
import numpy as np
from torch.nn import functional as F

from utils import utils
from os import path
import boto3
import os
input_sz = 320


def predict(model,input,bg):

	image = cv2.imread(input)
	X, pad_up, pad_left, h_new, w_new = utils.preprocessing(image, expected_size=input_sz, pad_value=0)
	
	h,w  = image.shape[:2]
	
	if bg is not None:
		BACKGROUND = cv2.imread(bg)[...,::-1]
		BACKGROUND = cv2.resize(BACKGROUND, (w,h), interpolation=cv2.INTER_LINEAR)
		KERNEL_SZ = 25
		SIGMA = 0
	else:
		COLOR1 = [255, 0, 0]
		COLOR2 = [0, 0, 255]
	
	with torch.no_grad():
		mask = model(X)
		mask = mask[..., pad_up: pad_up+h_new, pad_left: pad_left+w_new]
		mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=True)
		mask = F.softmax(mask, dim=1)
		mask = mask[0,1,...].numpy()
	
	if bg is None:
		image_alpha = utils.draw_matting(image, mask)
		# image_alpha = utils.draw_transperency(image, mask, COLOR1, COLOR2)
	
	else:
		image_alpha = utils.draw_fore_to_back(image, mask, BACKGROUND, kernel_sz=KERNEL_SZ, sigma=SIGMA)
	
	print(image_alpha.shape)
	
	return image_alpha
	
