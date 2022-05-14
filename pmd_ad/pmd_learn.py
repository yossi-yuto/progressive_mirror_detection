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

import eval

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

X_train = os.listdir(x_train_dir) # 訓練画像
Y_train = os.listdir(y_train_dir) # マスク画像
Y_edge_train = os.listdir(y_edge_train_dir) # マスクエッジ画像


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

""" (3)モデルとパラメータ探索アルゴリズムの設定 """
model     = PMD().cuda(device_ids[0])                # モデル
optimizer = optim.SGD(model.parameters(),lr=0.05) # パラメータ探索アルゴリズム(確率的勾配降下法 + 学習率lr=0.05を適用)
criterion = PMD_LOSS()               # 損失関数

""" (4) モデル学習 """
data_size  = len(X_train)           # データのサイズ
mini_batch = int(data_size * 3/4)   # ミニバッチサイズ(全データの3/4を学習に利用)
repeat = 150                       # エポック数

""" 学習中の評価用リスト"""
epoch_list = []
loss_list = []
iou_list = []


""" 学習 """
for epoch in range(repeat):

    print("training...")
    # permutation = 渡した引数の数値をシャッフル
    dx = np.random.permutation(data_size)

    dx_train = dx[:mini_batch] # 訓練データ
    dx_eval = dx[mini_batch:] # 評価用データ

    start = time.time()
    count = 0
    for idx in dx_train:

        img = Image.open(os.path.join(x_train_dir,X_train[idx])) # 訓練画像はそのまま読み込む
        if img.mode != 'RGB':
            img = img.convert('RGB')
            print("{} is a gray image.".format(idx))
        w, h = img.size

        img_label = Image.open(os.path.join(y_train_dir,(os.path.splitext(os.path.basename(X_train[idx]))[0] + '.png'))) # maskの拡張子がpngのためX_trainと同じbasenameを持つファイルを読み込む

        img_edge_label = Image.open(os.path.join(y_edge_train_dir,(os.path.splitext(os.path.basename(X_train[idx]))[0] + '.png')))
        
        # 訓練データとラベル画像
        img_var = Variable(img_transform(img).unsqueeze(0)).cuda() #訓練画像の配列（説明変数）
        target = Variable(torch.unsqueeze(img_label_transform(img_label),0)).cuda() # マスク画像の配列（正解値）
        target_edge = Variable(torch.unsqueeze(img_label_transform(img_edge_label),0)).cuda() # mask_edgeの画像配列

        # モデルのforward関数を用いた準伝播の予測→出力値算出
        pred = model(img_var) # layer4_predict, layer3_predict, layer2_predict, layer1_predict, layer0_edge, final_predict 

        # 上記出力値(output)と教師データ(target)を損失関数に渡し、損失関数を計算
        loss = criterion(pred[0], pred[1], pred[2], pred[3], pred[4],pred[5], target, target_edge)

        # 勾配を初期化
        optimizer.zero_grad()
        
        # 損失関数の値から勾配を求め誤差逆伝播による学習実行
        loss.backward()

        # 学習結果に基づきパラメータを更新
        optimizer.step()

    elasped_time = time.time() - start
    print("elapsed_time:{0}".format(elasped_time) + "[sec]")
    
    print("評価関数の計算...")

    # 評価関数(IoU)
    '''検証データを使用してIoUを算出'''
    iou = 0
    len = len(dx_eval)
    for idx in dx_eval:
        img = Image.open(os.path.join(x_train_dir,X_train[idx])) 
        img_lavel = Image.open(os.path.join(y_train_dir,(os.path.splitext(os.path.basename(X_train[idx]))[0] + '.png')))
        img_var = Variable(img_transform(img).unsqueeze(0)).cuda() #訓練画像の配列（説明変数）
        target = Variable(torch.unsqueeze(img_label_transform(img_label),0)).cuda() # maskの拡張子がpngのためX_trainと同じbasenameを持つファイルを読み込む
        
        output = model(img_var)
        iou += iou_binary(output[5].view(-1), target.view(-1)) # output[5] は　final_map

    # 評価関数
    print('\nEpoch: {}'.format(epoch+1))
    print("IoU:{}".format(iou / len)) # IoU
    print("loss:{loss}") # Loss
    eval.append_epoch(epoch_list,epoch)
    eval.append_loss(epoch_list, loss_list)

    '''
    問題点２：評価関数にF1_scoreとMAEを実装できていない。
    本来ならば評価関数にはF1_scoreとMAEを使用
    '''

"""学習状況をグラフで作成"""
eval.plot_IoU(epoch_list, iou_list)
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