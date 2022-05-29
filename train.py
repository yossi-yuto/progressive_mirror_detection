from model.pmd import PMD
import torch
from torch import optim
import torch.nn as nn

import pdb

import os
from torch.autograd import Variable
from torchvision import transforms

from loss.pmd_loss import PMD_LOSS
from torchmetrics import JaccardIndex

import pickle
import dataset
import matplotlib.pyplot as plt
from tqdm import tqdm

import argparse
import earlystop

'''コマンドライン引数'''
parser = argparse.ArgumentParser()
parser.add_argument('-epochs', type=int, default=150, help='The number of epoch')
parser.add_argument('-batch_size', type=int, default=5, help='Batch Size')
parser.add_argument('-resize', type=int, default=416 , help='input image resize')
parser.add_argument('-model_path', type=str, default='pmd_model', help='Path : save parameters of best model')
parser.add_argument('-checkpoint_dir', type=str, default='checkpoint', help='Directory : save parameter')
opt = parser.parse_args()


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
    transforms.Resize((opt.resize, opt.resize)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# mask,mask_edge
img_label_transform = transforms.Compose({
    transforms.Resize((opt.resize,opt.resize)),
    transforms.ToTensor()
})

# image data set
dataset = dataset.PmdDataset(x_train_dir,y_train_dir,y_edge_train_dir,img_transform,img_label_transform)

train_size = int(dataset.__len__() * 0.8) # train data 8割
val_size   = dataset.__len__() - train_size # validation data 2割

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

""" (3)モデルとパラメータ探索アルゴリズムの設定 """
model = PMD().cuda(device_ids[0])                # モデル
model.train() #訓練モード

#optimizer = optim.Adam(model.parameters(),lr=1e-3,momentum=0.9,weight_decay=5e-4) # パラメータ探索アルゴリズム(確率的勾配降下法 + 学習率lr=0.05を適用)
optimizer = optim.Adam(model.parameters())
#scheduler_poly_lr_decay = PolynomialLRDecay(optimizer, max_decay_steps=100, end_learning_rate=0.0001, power=0.9) # 学習率のスケジューラ
criterion = PMD_LOSS()                # 損失関数
jaccard   = JaccardIndex(num_classes=2) # 評価指標：IoU  binary→ 2クラス 


""" 学習中の評価（グラフ用）リスト"""
train_loss_list = []
val_loss_list   = []
train_iou_list  = []
val_iou_list    = []

# dataloader
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=opt.batch_size,shuffle=True,num_workers=2, pin_memory= True)
val_loader   = torch.utils.data.DataLoader(val_dataset,batch_size=1,num_workers=2, pin_memory = True) 

""" 学習 """
es = earlystop.EarlyStopping(verbose=True, path=os.path.join(opt.checkpoint_dir, opt.model_path)) # early stoppingをするためのメソッド

for epoch in range(opt.epochs):

    print('\nEpoch: {}'.format(epoch+1))

    torch.backends.cudnn.benchmark = True

    # 訓練時評価用
    train_loss = 0
    train_iou = 0
    
    for image, mask, edge in tqdm(train_loader): # すべてのデータセット分だけ実行
        
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
        
        # loss,jaccardIndex
        preds = (torch.nn.functional.sigmoid(pred[5]) > 0.5).long()
        train_iou  += jaccard(preds.cpu(),target.int().cpu())
        train_loss += loss.item()

    # 訓練データの計算
    train_iou /= len(train_loader)
    train_loss /= len(train_loader)
    
    print(" Train_Loss:{}, Train_IoU:{}".format(train_loss,train_iou))
    train_loss_list.append(train_loss)
    train_iou_list.append(train_iou)
    
    '''検証データを使用'''
    with torch.no_grad(): # 勾配の計算しない
        val_iou = 0
        val_loss = 0
        for image, mask, edge in tqdm(val_loader):                   
            pred = model(image.cuda())
            mask = mask.cuda()
            edge = edge.cuda()
            loss = criterion(pred[0], pred[1], pred[2], pred[3], pred[4],pred[5], mask, edge)   # pred[5] はfinal_map     
            
            preds = (torch.nn.functional.sigmoid(pred[5]) > 0.5).long()  # 閾値 0.5
            val_iou += jaccard(preds.cpu(), mask.int().cpu()) 
            val_loss += loss.item()

        # 評価データの計算
        val_iou /= len(val_loader) 
        val_loss /= len(val_loader)
        print("Valid_Loss:{}, Valid_IoU:{}, ".format(val_loss, val_iou)) 

        val_loss_list.append(val_loss)
        val_iou_list.append(val_iou)

        es(val_loss, model)
        if es.early_stop:
            print("Early Stopping!")
            break

    if epoch % 5 == 0:
        with open('train_data.pickle','wb') as w: 
            pickle.dump(train_loss_list, w)
            pickle.dump(val_loss_list, w)
            pickle.dump(train_iou_list, w)
            pickle.dump(val_iou_list, w)

pdb.set_trace()

'''
問題点２：評価関数にF1_scoreとMAEを実装できていない。
本来ならば評価関数にはF1_scoreとMAEを使用
https://canary.discord.com/7a0b734a-9633-4108-830e-8598cdbde225   
'''

""" (5)モデルの結果を出力 """
#torch.save(model.state_dict(), "pmd_model.pth")    # モデル保存する場合
#model.load_state_dict(torch.load("pmd_model.pth")) # モデルを呼び出す場合



