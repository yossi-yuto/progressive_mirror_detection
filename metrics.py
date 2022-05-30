
import torch
import torch.nn as nn
from torchvision import transforms
from torchmetrics import JaccardIndex

from PIL import Image
import os
import pdb
from tqdm import tqdm
        

def testmetrics(pred_dir, test_mask_dir):
    '''
    Arg:
        テスト画像における推論結果とマスク画像を比較し精度を求める
        pred_dir：推論結果のマスク画像が入ったディレクトリのパス
        test_mask_dir：テスト用のマスク画像が入ったディレクトリのパス
    
    '''
    
    to_tensor = transforms.ToTensor()

    jaccard = JaccardIndex(num_classes=2)

    img_list = os.listdir(pred_dir)

    score = 0 # 評価値が格納
    img_len = len(img_list) # 推論結果の総数

    for img_name in tqdm(img_list):
        
        img_array = to_tensor(Image.open(os.path.join(pred_dir,img_name)).convert('1'))
        img_test_array = to_tensor(Image.open(os.path.join(test_mask_dir, img_name)).convert('1'))

        score += jaccard(img_array, img_test_array.int()).item()

    return score / img_len



def main():

    # パスの指定
    pred_dir = './result_sub/PMD_test_COCO'
    test_mask_dir = './PMD/test/COCO/mask'

    # スコアの計算と表示
    score = testmetrics(pred_dir, test_mask_dir)

    print(f"推論結果と正解マスク画像のmIoUスコアは {score} です。")

if __name__ == '__main__':
    main()




