#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import os, cv2, torch
import numpy as np

#------------------------------------------------------------------------------
#  Preprocessing
#------------------------------------------------------------------------------
mean = np.array([0.485, 0.456, 0.406])[None,None,:]
std = np.array([0.229, 0.224, 0.225])[None,None,:]

def resize_image(image, expected_size, pad_value, ret_params=False, mode=cv2.INTER_LINEAR):
	"""
	image (ndarray) with either shape of [H,W,3] for RGB or [H,W] for grayscale.
	Padding is added so that the content of image is in the center.
	"""
	h, w = image.shape[:2]
	if w>h:
		w_new = int(expected_size)
		h_new = int(h * w_new / w)
		image = cv2.resize(image, (w_new, h_new), interpolation=mode)

		pad_up = (w_new - h_new) // 2
		pad_down = w_new - h_new - pad_up
		if len(image.shape)==3:
			pad_width = ((pad_up, pad_down), (0,0), (0,0))
			constant_values=((pad_value, pad_value), (0,0), (0,0))
		elif len(image.shape)==2:
			pad_width = ((pad_up, pad_down), (0,0))
			constant_values=((pad_value, pad_value), (0,0))

		image = np.pad(
			image,
			pad_width=pad_width,
			mode="constant",
			constant_values=constant_values,
		)
		if ret_params:
			return image, pad_up, 0, h_new, w_new
		else:
			return image

	elif w<h:
		h_new = int(expected_size)
		w_new = int(w * h_new / h)
		image = cv2.resize(image, (w_new, h_new), interpolation=mode)

		pad_left = (h_new - w_new) // 2
		pad_right = h_new - w_new - pad_left
		if len(image.shape)==3:
			pad_width = ((0,0), (pad_left, pad_right), (0,0))
			constant_values=((0,0), (pad_value, pad_value), (0,0))
		elif len(image.shape)==2:
			pad_width = ((0,0), (pad_left, pad_right))
			constant_values=((0,0), (pad_value, pad_value))

		image = np.pad(
			image,
			pad_width=pad_width,
			mode="constant",
			constant_values=constant_values,
		)
		if ret_params:
			return image, 0, pad_left, h_new, w_new
		else:
			return image

	else:
		image = cv2.resize(image, (expected_size, expected_size), interpolation=mode)
		if ret_params:
			return image, 0, 0, expected_size, expected_size
		else:
			return image

def preprocessing(image, expected_size=224, pad_value=0):
	image, pad_up, pad_left, h_new, w_new = resize_image(image, expected_size, pad_value, ret_params=True)
	image = image.astype(np.float32) / 255.0
	image = (image - mean) / std
	X = np.transpose(image, axes=(2, 0, 1))
	X = np.expand_dims(X, axis=0)
	X = torch.tensor(X, dtype=torch.float32)
	return X, pad_up, pad_left, h_new, w_new


#------------------------------------------------------------------------------
#  Draw image with transperency
#------------------------------------------------------------------------------
def draw_transperency(image, mask, color_f, color_b):
	"""
	image (np.uint8)
	mask  (np.float32) range from 0 to 1 
	"""
	mask = mask.round()
	alpha = np.zeros_like(image, dtype=np.uint8)
	alpha[mask==1, :] = color_f
	alpha[mask==0, :] = color_b
	image_alpha = cv2.add(image, alpha)
	return image_alpha


#------------------------------------------------------------------------------
#   Draw matting
#------------------------------------------------------------------------------
def draw_matting(image, mask):
	"""
	image (np.uint8)
	mask  (np.float32) range from 0 to 1 
	"""
	mask = 255*(1.0-mask)
	mask = np.expand_dims(mask, axis=2)
	mask = np.tile(mask, (1,1,3))
	mask = mask.astype(np.uint8)
	image_matting = cv2.add(image, mask)
	return image_matting


#------------------------------------------------------------------------------
#  Draw foreground pasted into background
#------------------------------------------------------------------------------
def draw_fore_to_back(image, mask, background, kernel_sz=13, sigma=0):
	"""
	image (np.uint8)
	mask  (np.float32) range from 0 to 1 
	"""
	mask_filtered = cv2.GaussianBlur(mask, (kernel_sz, kernel_sz), sigma)
	mask_filtered = np.expand_dims(mask_filtered, axis=2)
	mask_filtered = np.tile(mask_filtered, (1,1,3))
	image_alpha = image*mask_filtered + background*(1-mask_filtered)
	return image_alpha.astype(np.uint8)