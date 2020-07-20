# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:29:57 2020

@author: aditya
"""

import pickle

token_path = 'datasets/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt'

def load_doc(path):
    file = open(path, 'r')
    doc = file.read()
    return doc

doc = load_doc(token_path)
    
def load_captions(doc):
    descriptions = dict()
    try:
        for line in doc.split('\n'):
            tokens = line.split('\t')
            image_id, img_desc = tokens[0], tokens[1:]
            
            image_id = image_id.split('.')[0]
            
            img_desc = ' '.join(img_desc).lower()
            
            if image_id in descriptions:
                descriptions[image_id].append(img_desc)
            else:
                descriptions[image_id] = list()
                descriptions[image_id].append(img_desc)
        return descriptions        
    except Exception as e:
        print('Got Exception: '+e)        

descriptions = load_captions(load_doc(token_path))
print(len(descriptions))    
'''f = open("descriptions.txt","w")
f.write(str(descriptions))
f.close()'''