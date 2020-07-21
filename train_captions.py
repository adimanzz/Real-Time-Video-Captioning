# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 19:58:48 2020

@author: aditya
"""
#import packages
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import keras
from keras.models import Model, Sequential
from keras.layers import LSTM
from keras.layers import Flatten, Dense, Dropout, Activation, Input ,BatchNormalization
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras import Input
from keras.layers.merge import add
from keras.models import Model 
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from keras.optimizers import Adam
from model import CaptionNet


token_path  = 'datasets/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt'

doc = open(token_path)
docs = doc.read()

f = open('datasets/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt','r')
train_names = f.read()
train = []

for line in train_names.split('\n'):
    name = line.split('.')[0]
    train.append(name)

#Create a dictionary to which includes the image id and the corresponding captions
train_descriptions = dict()        

for line in docs.split('\n'):
    tokens = line.split('\t')
    image_id, img_desc = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    
    if image_id in train and len(image_id) > 1:
        
        desc = 'startseq '+' '.join(img_desc)+' endseq'
        
        train_descriptions[image_id] = []
        train_descriptions[image_id].append(desc)
        
    
    else:
        continue
print(train_descriptions)    
    
all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap.lower())
print(len(all_train_captions))   

# Remove words that occur less than 5 times to make the model robust to outliers
# Also it'll significantly reduce the trainable parameters and hence make predictions faster
word_count_threshold = 5
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))

ixtoword = {}
wordtoix = {}

#pickle.dump(wordtoix,open('wordtoix_28.pkl','wb')) 2 Refers to the Threshold and 8 for Flickr8k
#pickle.dump(ixtoword,open('ixtoword_28.pkl','wb'))

#pickle.dump(wordtoix,open('wordtoix_58.pkl','wb')) 5 Refers to the Threshold and 8 for Flickr8k
#pickle.dump(ixtoword,open('ixtoword_58.pkl','wb'))

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1
    
VOCAB_SIZE = len(ixtoword) + 1   
print(VOCAB_SIZE)

        
#Max Length
def maxLength(dictionary):
    max_length = 0
    avg_length = 0
    n = 0
    for k,v in dictionary.items():
        length = len(max(v, key = len).split())
        avg_length += length
        n += 1
        if length > max_length:
            max_length = length
    return max_length, avg_length/n

MAX_LENGTH, avg_length = maxLength(train_descriptions)
print(MAX_LENGTH, avg_length)
#MAX_LENGTH = 80


#Embedding
glove_path = "Glove\glove.6B.300d.txt"    

embeddings_index = {} # empty dictionary
f = open(glove_path, encoding="utf-8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 300

embedding_matrix = np.zeros((VOCAB_SIZE, embedding_dim))

for word, i in wordtoix.items():
    embedding_vector = embeddings_index.get(word)   
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
    
print(embedding_matrix.shape)

def data_generator(descriptions, img_features, wordtoix, MAX_LENGTH, VOCAB_SIZE, batch_size):
    x1, x2, y = list(), list(), list()
    batch = 0
    
    while True:
        for key, caption_list in descriptions.items():
            
            batch += 1
            
            features = img_features[key]
            
            for caption in caption_list:
                seq = [wordtoix[word] for word in caption.split(' ') if word in wordtoix]
                
                for i in range((len(seq))):
                    in_seq, out_seq = seq[:i], seq[i] 
                    # for a given sequence the model will predict the next word
                    in_seq = pad_sequences([in_seq], maxlen = MAX_LENGTH)[0]
                    out_seq = to_categorical([out_seq], num_classes = VOCAB_SIZE)[0]
                    
                    x1.append(features)
                    x2.append(in_seq)
                    y.append(out_seq)
                    
                    if batch == batch_size:
                        yield [[np.array(x1), np.array(x2)], np.array(y)]
                        x1, x2, y = list(), list(), list()
                        batch = 0


#Model
model = CaptionNet(MAX_LENGTH, VOCAB_SIZE, embedding_dim)
model = model.forward()

model.layers[1].set_weights([embedding_matrix])
model.layers[1].trainable = False


opt = Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.00005)

model.compile(optimizer = opt, loss = 'categorical_crossentropy')

#Training
img_features = pd.read_pickle('image_features/img_features.pkl')
epochs = 10
batch_size = 6
steps = len(train_descriptions)//batch_size


for i in range(epochs):
    generator = data_generator(train_descriptions, img_features, wordtoix, MAX_LENGTH, VOCAB_SIZE, batch_size)
    model.fit_generator(generator, steps_per_epoch = steps, epochs = 1, verbose = 1)
    print('Epoch: {}/{}'.format(str(i+1),str(epochs)))
    
        

# lr = 0.001, decay = 0.00001, epochs = 25, bs = 6, attention, bidirect
#model.save('Model/model2_bi.h5')

# lr = 0.001, decay = 0.00001, epochs = 25, bs = 6, attention, bidirect, 5 sent each
#model.save('Model/model3_bi.h5')

# lr = 0.001, decay = 0.00005, epochs = 15, bs = 6, attention, bidirect, 5 sent each
#model.save('Model/model6_bi.h5') #Best

# lr = 0.001, decay = 0.00005, epochs = 10, bs = 6, attention, no bidirect, Flickr 30k, len=40
#model.save('Model/main_model1.h5')

#### Testing
import math
from math import log
import cv2

from attention import Attention
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

model_encoder = InceptionV3(weights = 'imagenet')
model_encoder = Model(model_encoder.input, model_encoder.layers[-2].output)
target_size = (299,299)

model2 = keras.models.load_model('Model/model6_bi.h5',
                                custom_objects={'Attention':Attention})

def preprocess_img(image_path, target_size):
    img = image.load_img(image_path, target_size)
    img = image.img_to_array(img)
    img = cv2.resize(img, target_size)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis = 0)
    return img




def decode(features):
    seq_pred = 'startseq'
    for i in range(MAX_LENGTH):
        sequence = [wordtoix[word] for word in seq_pred.split() if word in wordtoix]
        sequence = pad_sequences([sequence], maxlen = MAX_LENGTH, padding='post')
        pred = model.predict([features, sequence])
        pred = np.argmax(pred)
        word = ixtoword[pred]
        seq_pred += ' ' + word
        if word == 'endseq':
            break
        
    final = seq_pred.split()
    final = final[1: -1]
    final = ' '.join(final)
    return final  

    
img_path = 'test/basketball.jpg'
img = preprocess_img(img_path, target_size)
features = model_encoder.predict(img)            
pred_caption = decode(features)        
print(pred_caption)        






# Work Required

'''
def beam_search_decoder(features, k):
    seq_pred = 'startseq'
    seqq = []
    for i in range(MAX_LENGTH):
        sequence = [wordtoix[word] for word in seq_pred.split() if word in wordtoix]
        sequence = pad_sequences([sequence], maxlen = MAX_LENGTH)
        pred = model.predict([features, sequence])
        
        sequences = [[list(), 0.0]]
        for row in pred:
            all_candidates = list()
            for i in range(len(sequences)):
                seq, score = sequences[i]
                
                for j in range(len(row)):
                    candidate = [seq+[j], score - math.log1p(row[j])]
                    all_candidates.append(candidate)
                    
            ordered = sorted(all_candidates, key = lambda x: x[1])
            #select k best 
            sequences = ordered[:k]
        idx = sequences[0][0][0]
        word = ixtoword[idx]
        seq_pred += ' ' + word
        if word == 'endseq':
            break
    final = seq_pred.split()
    final = final[1: -1]
    final = ' '.join(final)
    return final 

pred_caption_beam = beam_search_decoder(features, 3)
print(pred_caption_beam)


 '''  
        
        
        
        
        
        
        
        
        
        
        
        

