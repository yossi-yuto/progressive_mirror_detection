import matplotlib.pyplot as plt

def plot_iou(epoch_list , iou_list):

    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.plot(iou_list, epoch_list)
    plt.savefig("IoU.png")

def plot_loss(epoch_list , loss_list):

    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.plot(loss_list, epoch_list)
    plt.savefig("loss.png")
