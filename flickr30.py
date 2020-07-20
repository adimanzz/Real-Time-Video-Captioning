# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 18:16:12 2020

@author: aditya
"""

import numpy as np
import pandas as pd
import string

df = pd.read_csv('flickr30k_images/results.csv', sep = '|')
df.columns = ['image','caption_num', 'caption']
#Since only one caption was missing filled it manually based on it's other captions
df['caption'] = df['caption'].fillna('A white , black , and brown dog runs in a field')

train = df['image'].tolist()
train = list(dict.fromkeys(train))
print(len(train))

train_descriptions = dict()

for x, y in df[['image','caption']].iterrows():
    name, desc = y[0], y[1]
    
    desc = 'startseq ' + desc.lower() + ' endseq'  
    # Remove punctuations
    #desc = desc.translate(str.maketrans('','',string.punctuation))
    
    if len(desc.split()) <= 40:
    
        if name not in train_descriptions:
            train_descriptions[name] = list()
            train_descriptions[name].append(desc)
        else:
            #continue
            train_descriptions[name].append(desc)
    else:
        continue        
            
print(len(train_descriptions))       
        
        
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


image_path = 'Flickr_Data/Flickr_Data/Images/667626_18933d713e.jpg'
temp = image.load_img('image3.jpg', target_size)
temp = image.img_to_array(temp)
temp = cv2.resize(temp, target_size)
temp = preprocess_input(temp)
temp = np.expand_dims(temp, axis = 0)
temp_pred = model.predict(temp)
#Model
model_encoder = InceptionV3(weights = 'imagenet')
model_encoder = Model(model_encoder.input, model_encoder.layers[-2].output)
target_size = (299,299)

def preprocess_img(image_path, target_size):
    img = image.load_img(image_path, target_size)
    img = image.img_to_array(img)
    img = cv2.resize(img, target_size)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis = 0)
    return img
    

def extract_features(image_name):    
    try:
        image_features = dict()
        for name in tqdm(image_name):
            if len(name) > 1:
                img = image.load_img('flickr30k_images/flickr30k_images/{}'.format(name),target_size)
                img = image.img_to_array(img)
                img = cv2.resize(img, target_size)
                img = preprocess_input(img)
                img = np.expand_dims(img, axis = 0)
                ft_vec = model_encoder.predict(img)
                image_features[name] = np.reshape(ft_vec, ft_vec.shape[1])
            else:
                break
        return image_features    
    except Exception as e:
        print('Got Exception: ',e)   
        
#save features
#img_features = extract_features(train)
#pickle.dump(img_features,open('img_features.pkl','wb'))        



        