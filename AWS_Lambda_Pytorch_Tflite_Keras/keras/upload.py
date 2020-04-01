try:
  import unzip_requirements
except ImportError:
  pass

import os
import json
import time

import boto3
import tensorflow as tf
import numpy as np
import keras
import boto3
from os import path
import cv2
from keras.models import load_model
import uuid


bucket_name = 'keras-background-model'
file_name = 'main_model.hdf5'
test_image = 'test.jpg'
model_path = '/tmp/'+file_name

graph = tf.get_default_graph()


def downloadFromS3(strBucket,strKey,strFile):
    s3_client = boto3.client('s3')
    s3_client.download_file(strBucket, strKey, strFile)

def get_file_from_s3_to_tmp(file_name):

    temp_path = '/tmp/'+file_name

    if not path.exists(temp_path):
        downloadFromS3(bucket_name,file_name,temp_path)

def download_files():
    get_file_from_s3_to_tmp(file_name)
    get_file_from_s3_to_tmp(test_image)

def s3_upload_image(filename):

    client = boto3.client('s3')
    client.upload_file(filename, bucket_name, str(uuid.uuid4())+".jpg")

def preprocess_selfi():

    img = cv2.imread('/tmp/'+test_image)
    h,w = img.shape[0],img.shape[1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(224, 224))
    img = img/255.0

    return img, h, w


def pipeline():

    download_files()

    img, h, w = preprocess_selfi()
    model = load_model(model_path, compile=False)

    with graph.as_default():
        prediction = model.predict(img[None, :, :, :])

    prediction = prediction.reshape((224,224, -1))

    prediction = cv2.resize(prediction[:, :, 1], dsize=(w, h))
    prediction = prediction*255.0
    prediction = prediction.astype('uint8')

    prediction[prediction>0.5*255] = 255
    prediction[prediction<0.5*255] = 0

    cv2.imwrite('/tmp/prediction.jpg',prediction)
    s3_upload_image('/tmp/prediction.jpg')

def uploadHandler(event, context):

    if event['httpMethod'] == "POST":
        body = {
            "message": "Success, Go Serverless!"
        }

        pipeline()
        response = {
            "statusCode": 200,
            "body": json.dumps(body)
        }

        return response

    else:
        return {
            'statusCode': 200,
            'body': "GET done"
            }
 
