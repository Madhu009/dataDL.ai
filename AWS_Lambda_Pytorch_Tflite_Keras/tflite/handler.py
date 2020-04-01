import json
import datetime
import numpy as np
import os
import tflite_runtime.interpreter as tflite
import boto3
from os import path
import cv2
import base64
import uuid
import io
import pickle
from io import BytesIO

file_name = 'mobilenet_v1_1.0_224_quant.tflite'
test_image = 'test.jpg'
labels_file = 'labels_mobilenet_quant_v1_224.txt'
bucket_name = 'tflite-image-classification'
model_path = '/tmp/'+file_name


def downloadFromS3(strBucket,strKey,strFile):
    s3_client = boto3.client('s3')
    s3_client.download_file(strBucket, strKey, strFile)

def get_file_from_s3_to_tmp(file_name):

	temp_path = '/tmp/'+file_name

	if not path.exists(temp_path):
		downloadFromS3(bucket_name,file_name,temp_path)

def download_files():
	get_file_from_s3_to_tmp(file_name)
	get_file_from_s3_to_tmp(labels_file)
	get_file_from_s3_to_tmp(test_image)

def preprocess_selfi():

	img = cv2.imread('/tmp/'+test_image)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	img = cv2.resize(img, dsize=(224, 224))
	img = np.expand_dims(img, axis=0)

	return img

def get_labels():
	with open("/tmp/"+labels_file,'r') as file:
		return [line.strip() for line in file.readlines()]

def classify(event, context):

	version_dict = {
		"numpy":np.__version__,
		"cv2":cv2.__version__
		}

	if event['httpMethod'] == "POST":
		
		download_files()

		interpreter_inf = tflite.Interpreter(model_path)

		interpreter_inf.allocate_tensors()
		input_details = interpreter_inf.get_input_details()
		output_details = interpreter_inf.get_output_details()
		input_shape = input_details[0]['shape']

		img = preprocess_selfi()

		interpreter_inf.set_tensor(input_details[0]['index'], img)
		interpreter_inf.invoke()
		output_data = interpreter_inf.get_tensor(output_details[0]['index'])
		output_idx = np.argmax(output_data[0])
		
		data_list = get_labels()
		predicted_label = data_list[output_idx]

		return {
			'statusCode': 200,
			'label': predicted_label
			}
	else:
		return {
			'statusCode': 200,
			'body': version_dict
			}






