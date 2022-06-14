import json 
import pickle
import numpy as np
import os

vid_train  = json.load(open('Charades_v1_train2.json'))['vid']
vid_test  = json.load(open('Charades_v1_test2.json'))['vid']
vid_val  = json.load(open('Charades_v1_val2.json'))['vid']
vid = vid_train+vid_test+vid_val
print('len(vid)',len(vid))

for id in vid:
    if not os.path.exists(f'../Charades_feature/{id}.npy'):
        print(f'{id}.npy not exists')
