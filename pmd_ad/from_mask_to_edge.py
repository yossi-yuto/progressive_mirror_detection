import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import pdb

'''
from_mask_to_edge.py
用途：
データセットのマスク画像からエッジの部分を抽出した画像を作成。
'''

DATA_DIR = './PMD'

# output path
outpath = os.path.join(DATA_DIR,  'train/edge') # フォルダ（Edge画像）
if not os.path.exists(outpath):
    os.mkdir(outpath)

# input path
DATA_DIR = './PMD'
y_train_dir = os.path.join(DATA_DIR, 'train/mask')
Y_train = os.listdir(y_train_dir) # マスク画像のパスを一括取得

# edge makiking
print("making now ...")
for filename in Y_train:

    

    filepath = os.path.join(y_train_dir,filename)

    image = cv2.imread(filepath)
    edge = cv2.Canny(image, 100,200) # 疑問点なぜ100,200なのかがわからない。
    cv2.imwrite(os.path.join(outpath,filename), edge)
print("finished!")






