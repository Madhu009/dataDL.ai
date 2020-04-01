try:
	import unzip_requirements
except ImportError:
	pass

import json
import torch
import torchvision
import numpy as np
import cv2
import os
from model import UNet
import boto3
from os import path
from inference import *
import uuid
print(os.listdir("/opt/python/"))

bg = None

mybucket = 'testmedium'

#fetch the model from S3
def get_model():
	
	if path.exists("/tmp/UNet_MobileNetV2.pth"):
		file_path = os.path.join('/tmp/','UNet_MobileNetV2.pth')
	else:
		strKey = 'UNet_MobileNetV2.pth'
		strFile = '/tmp/UNet_MobileNetV2.pth'
		downloadFromS3(mybucket,strKey,strFile)
		file_path = os.path.join('/tmp/','UNet_MobileNetV2.pth')
	return file_path

#upload the output file to S3
def s3_upload_image(filename):

	client = boto3.client('s3')
	client.upload_file(filename, mybucket, str(uuid.uuid4())+".jpg")

#download files from S3
def downloadFromS3(strBucket,strKey,strFile):
	s3_client = boto3.client('s3')
	s3_client.download_file(strBucket, strKey, strFile)
	
#fetch the input image from S3	
def get_input():
	
	if path.exists("/tmp/input2.jpg"):
		file_path = os.path.join('/tmp/','input2.jpg')
	else:
		strKey = 'input2.jpg'
		strFile = '/tmp/input2.jpg'
		downloadFromS3(mybucket,strKey,strFile)
		file_path = os.path.join('/tmp/','input2.jpg')
	return file_path
	
def load_model():
	
	checkpoint = get_model()
	
	model = UNet(
		backbone="mobilenetv2",
		num_classes=2,
		pretrained_backbone=None
	)
	
	trained_dict = torch.load(checkpoint, map_location="cpu")['state_dict']
	model.load_state_dict(trained_dict, strict=False)
	model.eval()
	
	print("model is loaded")
	
	return model

#this is the entry point for this function
def lambda_handler(event, context):
	
	version_dict = {
		"numpy":np.__version__,
		"torch":torch.__version__,
		"torchvision":torchvision.__version__,
		"cv2":cv2.__version__
	}
	
	model = load_model()
	input = get_input()
	
	output = predict(model,input,None)
	
	filename = "/tmp/output.jpg"
	cv2.imwrite(filename,output)
	
	s3_upload_image(filename)
	
	return {
		'statusCode': 200,
		'body': version_dict
	}
