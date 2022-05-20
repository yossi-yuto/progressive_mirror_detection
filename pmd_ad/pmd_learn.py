from random import shuffle
from model.pmd import PMD
import torch
from torch import optim
import torch.nn as nn
from torch_poly_lr_decay import PolynomialLRDecay

import pdb

import os
import glob
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

import numpy as np
from loss.pmd_loss import PMD_LOSS
from loss.loss import iou_binary
import time

import pickle
import dataset
import matplotlib.pyplot as plt
from tqdm import tqdm

import gc

""" 各種設定 """
# GPU設定
device_ids = [0]
torch.cuda.set_device(device_ids[0])

# ファイルパス
DATA_DIR = './PMD'

x_train_dir = os.path.join(DATA_DIR, 'train/image')
y_train_dir = os.path.join(DATA_DIR, 'train/mask')
y_edge_train_dir = os.path.join(DATA_DIR, 'train/edge')

x_test_dir = os.path.join(DATA_DIR, 'test/image')
y_test_dir = os.path.join(DATA_DIR, 'test/mask')

'''(2) データ作成'''
# 入力画像の処理
# image
img_transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# mask,mask_edge
img_label_transform = transforms.Compose({
    transforms.Resize((416,416)),
    transforms.ToTensor()
})

# image data set
dataset = dataset.PmdDataset(x_train_dir,y_train_dir,y_edge_train_dir,img_transform,img_label_transform)

train_size = int(dataset.__len__() * 0.9) # train data 8割
val_size   = dataset.__len__() - train_size # validation data 2割

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])


""" (3)モデルとパラメータ探索アルゴリズムの設定 """
model = PMD().cuda(device_ids[0])                # モデル

#optimizer = optim.Adam(model.parameters(),lr=1e-3,momentum=0.9,weight_decay=5e-4) # パラメータ探索アルゴリズム(確率的勾配降下法 + 学習率lr=0.05を適用)
optimizer = optim.Adam(model.parameters())
#scheduler_poly_lr_decay = PolynomialLRDecay(optimizer, max_decay_steps=100, end_learning_rate=0.0001, power=0.9) # 学習率のスケジューラ
criterion = PMD_LOSS()               # 損失関数

""" (4) モデル学習 """
mini_batch = 5 # batch_size
repeat = 40   # エポック数

""" 学習中の評価（グラフ用）リスト"""
train_loss_list = []
val_loss_list   = []
train_iou_list  = []
val_iou_list    = []

model.layer0.requires_grad_ = False
model.layer1.requires_grad_ = False
model.layer2.requires_grad_ = False
model.layer3.requires_grad_ = False
model.layer4.requires_grad_ = False

start = time.time() # 処理時間の計測

""" 学習 """
print("training start...")
for epoch in range(repeat):

    print('\nEpoch: {}'.format(epoch+1))

    torch.backends.cudnn.benchmark = True

    # 訓練時評価用
    train_loss = 0
    train_iou = 0

    model.train() #訓練モード

    bar = tqdm(total = train_size)
    bar.set_description('Progress rate')    
    
    for image, mask, edge in tqdm(torch.utils.data.DataLoader(train_dataset,batch_size=mini_batch,shuffle=True,num_workers=2, pin_memory= True)): # すべてのデータセット分だけ実行
        
        # 勾配を初期化
        optimizer.zero_grad()
        
        # 訓練データとラベル画像
        img_var = image.cuda() #訓練画像の配列（説明変数）
        target = mask.cuda() # マスク画像の配列（正解値）
        target_edge = edge.cuda() # mask_edgeの画像配列

        # モデルのforward関数を用いた準伝播の予測→出力値算出
        pred = model(img_var) # layer4_predict, layer3_predict, layer2_predict, layer1_predict, layer0_edge, final_predict 
        
        # 上記出力値(output)と教師データ(target)を損失関数に渡し、損失関数を計算
        loss = criterion(pred[0], pred[1], pred[2], pred[3], pred[4],pred[5], target, target_edge)
        
        # 損失関数の値から勾配を求め誤差逆伝播による学習実行
        loss.backward()

        # 学習結果に基づきパラメータを更新
        optimizer.step()

        # loss
        preds = (torch.nn.functional.sigmoid(pred[5]) > 0.5).long()
        train_iou  += iou_binary(preds,target)
        train_loss += loss.item()


    # 評価ステップ
    train_loss /= len(train_dataset)
    train_iou /= len(train_dataset)
    print(" Train_Loss:{}, Train_IoU:{}".format(train_loss,train_iou))
    train_loss_list.append(train_loss)
    train_iou_list.append(train_iou)
    
    '''検証データを使用してIoUを算出'''
    model.eval() #評価モード
    val_iou = 0
    val_loss = 0
    with torch.no_grad(): # 勾配の計算しない
        for image, mask, edge in tqdm(torch.utils.data.DataLoader(val_dataset,batch_size=1,num_workers=2, pin_memory = True)):                   
            pred = model(image.cuda())
            mask = mask.cuda()
            edge = edge.cuda()
            loss = criterion(pred[0], pred[1], pred[2], pred[3], pred[4],pred[5], mask, edge)
            preds = (pred[5] > 0.5).long()  # 閾値 0.5
            val_iou += iou_binary(preds, mask) # output[5] は　final_map
            val_loss += loss.item()
        
    val_iou /= len(val_dataset) 
    val_loss /= len(val_dataset)
    print("Val_IoU:{}, Val_Loss:{}".format(val_iou, val_loss)) # IoU

    val_loss_list.append(val_loss)
    val_iou_list.append(val_iou)

    if epoch % 5 == 0:
        with open('train_data.pickle','wb') as w: # 訓練画像、マスク画像、 エッジ画像
            pickle.dump(train_loss_list, w)
            pickle.dump(val_loss_list, w)
            pickle.dump(train_iou_list, w)
            pickle.dump(val_iou_list, w)


    '''
    問題点２：評価関数にF1_scoreとMAEを実装できていない。
    本来ならば評価関数にはF1_scoreとMAEを使用
https://canary.discord.com/7a0b734a-9633-4108-830e-8598cdbde225    '''

pdb.set_trace()

""" (5)モデルの結果を出力 """
torch.save(model.state_dict(), "pmd_model.pth")    # モデル保存する場合
model.load_state_dict(torch.load("pmd_model.pth")) # モデルを呼び出す場合



""" (6) モデルの性能評価 """
# 学習したモデルの評価
'''
model.eval()
with torch.no_grad():
    pred_model  = model(X_test)              # テストデータでモデル推論
    pred_result = torch.argmax(pred_model,1) # 予測値

    # 正解率
    print(round(((Y_test == pred_result).sum()/len(pred_result)).item(),3))

'''
