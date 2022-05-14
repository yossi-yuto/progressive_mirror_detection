from loss.loss import lovasz_hinge

import torch
import torch.nn as nn
import pdb

class PMD_LOSS(nn.Module):
    def __init__(self,W_s = 1, W_b = 5, W_f = 2):
        super(PMD_LOSS, self).__init__()
        # weight
        self.W_s = W_s # 1
        self.W_b = W_b # 5
        self.W_f = W_f # 2
        self.BCE = nn.BCELoss()
        self.BCEwithLogitsLoss = nn.BCEWithLogitsLoss()
        

    def forward(self, input0,input1,input2,input3, input_edge,final_map, target,target_edge): # 追加でマスク画像から精製した境界の正解画像を追加　target_edge

        '''
        for i in range(0,len(input)):
            input[i] = input[i].squeeze(0) # layer0~5_predict squeeze(0)⇨(B,H,W)
        '''

        target = target.squeeze(0) # mask (B,H,W)
        target_edge = target_edge.squeeze(0) # mask_edge (B,H,W)

        # immeditate map
        lovasz_loss0 = lovasz_hinge(input0.squeeze(0), target)
        lovasz_loss1 = lovasz_hinge(input1.squeeze(0), target)
        lovasz_loss2 = lovasz_hinge(input2.squeeze(0), target)
        lovasz_loss3 = lovasz_hinge(input3.squeeze(0), target)

        #bce_loss = self.BCE(input_edge.view(-1), target_edge.view(-1)) # EDF module
        
        bce_loss = self.BCEwithLogitsLoss(input_edge.squeeze(0), target_edge)

        final_loss = lovasz_hinge(final_map, target)
        
        return self.W_s * lovasz_loss0 +  self.W_s * lovasz_loss1 + self.W_s * lovasz_loss2 + self.W_s * lovasz_loss3 +self.W_b * bce_loss + self.W_f * final_loss




