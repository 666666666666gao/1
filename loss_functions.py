import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()

    def forward(self, pred_gt, true_gt, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        pred_gt = F.sigmoid(pred_gt)       
        
        #flatten label and prediction tensors
        pred_gt = pred_gt.view(-1)
        true_gt = true_gt.view(-1)

        BCE_loss = F.binary_cross_entropy(pred_gt, true_gt, reduction='mean')
        
        return BCE_loss


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, pred_gt, true_gt, smooth=1e-8):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        pred_gt = F.sigmoid(pred_gt)       
        
        #flatten label and prediction tensors
        pred_gt = pred_gt.view(-1)   # target mask
        true_gt = true_gt.view(-1)  # true mask
        
        intersection = (pred_gt * true_gt).sum()                            
        dice = (2.*intersection + smooth)/(pred_gt.sum() + true_gt.sum() + smooth)  
        
        return 1 - dice
    
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, pred_gt, true_gt, smooth=1e-8):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        pred_gt = F.sigmoid(pred_gt)       
        
        #flatten label and prediction tensors
        pred_gt = pred_gt.view(-1)
        true_gt = true_gt.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (pred_gt * true_gt).sum()
        total = (pred_gt + true_gt).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

    

