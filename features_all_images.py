# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 16:07:38 2020

@author: aditya
"""

# Since many words in the vocab and descriptions are in the dev and test sets we'll include these too in the final model.
# This is once the model has been tuned on the train and test sets

token_path  = 'datasets/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt'

doc = open(token_path)
docs = doc.read()

f = open('datasets/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt','r')
train_names = f.read()

d = open('datasets/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.devImages.txt','r')
dev_names = d.read()

t = open('datasets/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt','r')
test_names = t.read()


train = []

for line in train_names.split('\n'):
    name = line.split('.')[0]
    train.append(name)

for line in dev_names.split('\n'):
    name = line.split('.')[0]
    train.append(name)

for line in test_names.split('\n'):
    name = line.split('.')[0]
    train.append(name)        

train_descriptions = dict()

        

for line in docs.split('\n'):
    tokens = line.split('\t')
    image_id, img_desc = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    
    if image_id in train and len(image_id) > 1:
        
        desc = 'startseq '+' '.join(img_desc)+' endseq'
        
        if image_id not in train_descriptions:
             train_descriptions[image_id] = []
             train_descriptions[image_id].append(desc)
        else:
             train_descriptions[image_id].append(desc)    
        
        
    else:
        continue
    
    
    
print(len(train_descriptions))    
    
    
all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap.lower())
print(len(all_train_captions))   

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
    for k,v in dictionary.items():
        length = len(max(v, key = len).split())
        if length > max_length:
            max_length = length
    return max_length

MAX_LENGTH = maxLength(train_descriptions)
print(MAX_LENGTH)

    
    
#Training
train_img_features = pd.read_pickle('image_features/train_image_extracted.pkl')
dev_img_features = pd.read_pickle('image_features/dev_image_extracted.pkl')
test_img_features = pd.read_pickle('image_features/test_image_extracted.pkl')

img_features = dict()
img_features.update(train_img_features)
img_features.update(dev_img_features)
img_features.update(test_img_features)

pickle.dump(train_image_extracted,open('image_features/img_features.pkl','wb'))
        