from cProfile import label
import pickle
import matplotlib.pyplot as plt
import pdb

def paint_gragh(train_list, val_list, x_label_name, y_label_name, savefig_name):
    plt.xlabel(x_label_name)
    plt.ylabel(y_label_name)
    plt.plot(train_list, label='Train')
    plt.plot(val_list, label='Valid')
    plt.legend()
    plt.savefig(f"{savefig_name}.jpg")
    plt.cla()

def paint_heatmap(tensor,file_name): # heatmapを描画するメソッド
    tensor = tensor.squeeze(0) #２次元座標に変化
    plt.imshow(tensor, vmin=0, vmax=1, cmap='Blues')
    plt.savefig(f"{file_name}")

with open('train_data.pickle', 'rb') as r:
    train_loss_list= pickle.load(r)  # img_array (N,3,H,W)
    val_loss_list  = pickle.load(r) # mask array (N,1,H,W)
    train_iou_list = pickle.load(r) # edge_array (N,1,H,W)
    val_iou_list   = pickle.load(r)

# グラフを描画する
paint_gragh(train_loss_list, val_loss_list, "epoch", "loss", "loss_50")
paint_gragh(train_iou_list, val_iou_list, "epoch", "JaccardIndex", "Jaccard_50")
