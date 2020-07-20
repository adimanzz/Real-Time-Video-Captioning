# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 11:50:28 2020

@author: aditya
"""
import tensorflow as tf
import keras
from keras.models import Model
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions

from keras.preprocessing import image
import pickle
import numpy as np
import argparse
import cv2
from tqdm import tqdm



#Model
model_encoder = InceptionV3(weights = 'imagenet')
# Remove the Final Layer since we need the hidden features and not the Label output
model_encoder = Model(model_encoder.input, model_encoder.layers[-2].output)
target_size = (299,299)

def preprocess_img(image_path, target_size):
    img = image.load_img(image_path, target_size)
    img = image.img_to_array(img)
    img = cv2.resize(img, target_size)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis = 0)
    return img
    

train_path = 'datasets/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt'
dev_path = 'datasets/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.devImages.txt'
test_path = 'datasets/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt'

def load_doc(path):
    f =  open(path,'r')
    doc = f.read()
    return doc

def extract_features(doc):    
    try:
        image_features = dict()
        for line in doc.split('\n'):
            image_id = line.split('.')[0]
            if len(image_id) > 1:
                img = image.load_img('Flickr_Data/Flickr_Data/Images/{}.jpg'.format(image_id),target_size)
                img = image.img_to_array(img)
                img = cv2.resize(img, target_size)
                img = preprocess_input(img)
                img = np.expand_dims(img, axis = 0)
                ft_vec = model_encoder.predict(img)
                image_features[image_id] = np.reshape(ft_vec, ft_vec.shape[1])
            else:
                break
        return image_features    
    except Exception as e:
        print('Got Exception: ',e)   
        
#save features
#train_image_extracted = extract_features(load_doc(train_path))
#pickle.dump(train_image_extracted,open('image_features/train_image_extracted.pkl','wb'))
    
#dev_image_extracted = extract_features(load_doc(dev_path))
#pickle.dump(dev_image_extracted, open('image_features/dev_image_extracted.pkl','wb'))        

#test_image_extracted = extract_features(load_doc(test_path))
#pickle.dump(test_image_extracted, open('image_features/test_image_extracted.pkl','wb'))


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
