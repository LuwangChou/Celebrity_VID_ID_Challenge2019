# -*- coding: utf-8 -*-
import pdb
import sys
import numpy as np
import pickle
import time
def load_pickle(fin):
    return pickle.load(fin, encoding='bytes')

start = time.process_time()
# face
face_path = './face_train_v2.pickle'
print('loading {}...'.format(face_path))
with open(face_path, 'rb') as fin:
    face_feats_dict = load_pickle(fin)

face_path = './face_val_v2.pickle'
print('loading {}...'.format(face_path))
with open(face_path, 'rb') as fin:
    face_feats_dict = load_pickle(fin)
face_time = time.process_time()
print("Read_Face_Time Used:{}".format(face_time-start))

# head
head_path = './head_train.pickle'
print('loading {}...'.format(head_path))
with open(head_path, 'rb') as fin:
    head_feats_dict = load_pickle(fin)
    
head_time = time.process_time()
print("Read_Head_Time Used:{}".format(head_time-face_time))

# # body
# body_path = './body_train.pickle'
# print('loading {}...'.format(body_path))
# with open(body_path, 'rb') as fin:
#     body_feats_dict = load_pickle(fin)

body_time = time.process_time()
print("Read_body_Time Used:{}".format(body_time-head_time))

# # audio
# audio_path = './audio_val.pickle'
# print('loading {}...'.format(audio_path))
# with open(audio_path, 'rb') as fin:
#     audio_feats_dict = load_pickle(fin)

audio_time = time.process_time()
print("Read_Audio_Time Used:{}".format(audio_time-body_time))
print("Load File Done!")
print("Total Time Used:{}".format(time.process_time()-start))
