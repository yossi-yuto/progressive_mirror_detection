from random import shuffle
from model.pmd import PMD
import torch
from torch import optim
import torch.nn as nn

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

import eval
from tqdm import tqdm

import gc

""" 各種設定 """
# GPU設定
device_ids = [0]
torch.cuda.set_device(device_ids[0])

# データ作成フラグ
make_flag = False

# ファイルパス
DATA_DIR = './PMD'

x_train_dir = os.path.join(DATA_DIR, 'train/image')
y_train_dir = os.path.join(DATA_DIR, 'train/mask')
y_edge_train_dir = os.path.join(DATA_DIR, 'train/edge')

x_test_dir = os.path.join(DATA_DIR, 'test/image')
y_test_dir = os.path.join(DATA_DIR, 'test/mask')

X_train = os.listdir(x_train_dir) # 訓練画像
Y_train = os.listdir(y_train_dir) # マスク画像
Y_edge_train = os.listdir(y_edge_train_dir) # マスクエッジ画像

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

# データ作成
if make_flag:
    train_list=[] # 訓練画像
    mask_list=[] # マスク画像
    edge_list=[] # エッジ画像

    for file in X_train:

        # 訓練画像
        train_img=Image.open(os.path.join(x_train_dir,file))
        if train_img.mode != 'RGB':
            train_img = train_img.convert('RGB')
            print("{} is a gray image.".format(file))
        train_list.append(img_transform(train_img))

        # マスク画像
        img_label = Image.open(os.path.join(y_train_dir,(os.path.splitext(os.path.basename(file))[0] + '.png')))
        img_label = img_label.convert("1") # グレースケールの画像
        mask_list.append(img_label_transform(img_label))

        # エッジ画像
        img_edge_label = Image.open(os.path.join(y_edge_train_dir,(os.path.splitext(os.path.basename(file))[0] + '.png')))
        img_edge_label = img_edge_label.convert("1") # グレースケールの画像
        edge_list.append(img_label_transform(img_edge_label))

    img_array = torch.stack(train_list)
    mask_array = torch.stack(mask_list)
    edge_array = torch.stack(edge_list)

    with open('train_data.pickle','wb') as w: # 訓練画像、マスク画像、 エッジ画像
        pickle.dump(img_array, w)
        pickle.dump(mask_array, w)
        pickle.dump(edge_array, w)

with open('train_data.pickle', 'rb') as r:
    img_tensor = pickle.load(r)  # img_array (N,3,H,W)
    mask_tensor = pickle.load(r) # mask array (N,1,H,W)
    edge_tensor = pickle.load(r) # edge_array (N,1,H,W)

# データセットの作成
data_size  = len(X_train)           # データのサイズ
train_split = int(data_size * 3/4)   # ミニバッチサイズ(全データの3/4を学習に利用)
Dataset = torch.utils.data.TensorDataset(img_tensor, mask_tensor, edge_tensor)
train_dataset, valid_dataset = torch.utils.data.random_split(Dataset,[train_split,data_size - train_split])

# Memory Delete
del img_tensor
del mask_tensor
del edge_tensor
del Dataset
gc.collect()

""" (3)モデルとパラメータ探索アルゴリズムの設定 """
model     = PMD().cuda(device_ids[0])                # モデル
optimizer = optim.SGD(model.parameters(),lr=1e-3,momentum=0.9,weight_decay=5e-4) # パラメータ探索アルゴリズム(確率的勾配降下法 + 学習率lr=0.05を適用)
criterion = PMD_LOSS()               # 損失関数

""" (4) モデル学習 """
mini_batch = 5 # batch_size
repeat = 150   # エポック数

""" 学習中の評価（グラフ用）リスト"""
epoch_list = [] 
loss_list = []
iou_list = []

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
    model.train() #訓練モード

    bar = tqdm(total = train_split)
    bar.set_description('Progress rate')    

    for image, mask, edge in tqdm(torch.utils.data.DataLoader(train_dataset,batch_size=mini_batch,shuffle=True,num_workers=os.cpu_count(), pin_memory= True)): # すべてのデータセット分だけ実行
        
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

        # タスクバー表示
        time.sleep(10)

        # 現在のIoUを表示
        '''IoUでは予測データが予測値がほぼ１になる問題がある。'''
        preds = (torch.nn.functional.sigmoid(pred[5]) > 0.5).long() # 閾値を0.5に設定
        print("現在のIoU : {}" .format(iou_binary(preds,target)))
        print("現在のLoss: {}\n".format(loss.item()) )

    elasped_time = time.time() - start
    print("elapsed_time:{} sec\n".format(elasped_time) )

    print("評価関数の計算...\n")
    # 評価関数(IoU)
    '''検証データを使用してIoUを算出'''
    model.eval() #評価モード
    iou = 0    
    with torch.no_grad(): # 勾配の計算しない
        for image, mask, edge in tqdm(torch.utils.data.DataLoader(valid_dataset,batch_size=1,shuffle=True,num_workers=2)):                   
            pred = (model(image.cuda())[5] > 0).long()
            mask = mask.cuda()
            preds = (torch.nn.functional.sigmoid(pred[5]) > 0.5).long()  # 閾値 0.5
            iou += iou_binary(preds, mask) # output[5] は　final_map
            
    mIoU = iou / len(valid_dataset) # IoUの平均値を求める。
    # 評価関数
    print("IoU:{}".format(mIoU)) # IoU
    print("loss:{}".format(loss.item())) # Loss
    epoch_list.append(epoch)
    loss_list.append(loss.item())
    iou_list.append(mIoU)


    '''
    問題点２：評価関数にF1_scoreとMAEを実装できていない。
    本来ならば評価関数にはF1_scoreとMAEを使用
    '''

"""学習状況をグラフで作成"""
eval.plot_iou(epoch_list, iou_list)
eval.plot_loss(epoch_list, loss_list)



""" (5)モデルの結果を出力 """
torch.save(model.state_dict(), "sample.model")    # モデル保存する場合
model.load_state_dict(torch.load("sample.model")) # モデルを呼び出す場合
pdb.set_trace()


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
