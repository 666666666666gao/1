import torch
import os 
import argparse
import pickle
import numpy as np
import torch.nn as nn
from torch import optim
from utils import CustomImageDataset, init_weights, DeepCrackDataset
from eval_metrics import f1_score, iou_score
from model import UNet_FCN, LMM_Net
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from efficientnet import EfficientCrackNet
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader
print(torch.cuda.is_available())
import warnings
warnings.filterwarnings("ignore")


# Testing model
def eval(args, test_dataloaders):
    
    if args.model_name == 'UNet':
        model = UNet_FCN(args = args, scaler=2).to(device)
        model.apply(init_weights)
    elif args.model_name == 'LMM_Net':
        model = LMM_Net().to(device)
    elif args.model_name == 'EfficientCrackNet':
        model = EfficientCrackNet().to(device)

    model.load_state_dict(torch.load(f'./saved_models/{args.model_name}/best_model_num_{args.run_num}.pt')['model_state_dict'])

    model.eval()
    f1_scores, recall_scores, precision_scores, iou_scores = 0.0, 0.0, 0.0, 0.0
    num_batch = 0.0
    # metrics: IoU, BF score, F1 score, precision, recall 

    for input_img, mask in test_dataloaders:
        input_img, mask = input_img.to(device), mask.to(device)
        with torch.no_grad():
            # Add batch to GPU
            output_mask = model(input_img)
            output_mask = (output_mask > 0.5).float()

            f1_s = f1_score(mask.cpu().numpy().flatten(), output_mask.cpu().numpy().flatten())
            recall = recall_score(mask.cpu().numpy().flatten(), output_mask.cpu().numpy().flatten())
            precision = precision_score(mask.cpu().numpy().flatten(), output_mask.cpu().numpy().flatten())
            cm = confusion_matrix(mask.cpu().numpy().flatten(), output_mask.cpu().numpy().flatten(), labels=[0, 1])

            intersection = np.diag(cm)
            ground_truth_set = cm.sum(axis=1)
            predicted_set = cm.sum(axis=0)
            union = ground_truth_set + predicted_set - intersection
            iou = intersection / union.astype(np.float32)
            iou_scores += np.mean(iou)

            f1_scores += f1_s
            recall_scores += recall
            precision_scores += precision
            num_batch += 1.0

    test_f1_score = (f1_scores/num_batch)
    test_recall_score = (recall_scores/num_batch)
    test_precision_score = (precision_scores/num_batch)
    test_miou_score = (iou_scores/num_batch)

    print(f'Test F1 Score is: {round(test_f1_score, 2)}')
    print(f'Test Recall Score is: {round(test_recall_score, 2)}')
    print(f'Test Precision Score is: {round(test_precision_score, 2)}')
    print(f'Test mIoU Score is: {round(test_miou_score, 2)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crack Segmentation Work')
    parser.add_argument('--data_dir', type=str, help='Main directory of input dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,help='Momentum in learning rate')
    parser.add_argument('--validate', type=bool, default=False, help='whether to validate the model or not')
    parser.add_argument('--model_name', type=str, help='algorithm')
    parser.add_argument('--epochs', type=int, help='Num of epochs')
    parser.add_argument('--alpha', type=float, help='Reduction factor in Loss function')
    parser.add_argument('--optim_w_decay', type=float, default=2e-4)
    parser.add_argument('--lr_decay', type=float, default=0.8)
    parser.add_argument('--num_epochs_decay', type=int, default=5)
    parser.add_argument('--data_name', type=str, help='Dataset to be used')
    parser.add_argument('--rgb', type=bool, help='Is image RGB or not')
    parser.add_argument('--run_num', type=str, help='run number')
    parser.add_argument('--half', type=bool, default=False, help='use half Model size or not')
    parser.add_argument('--augment', type=bool, default=False, help='whether augment dataset or not')

    args = parser.parse_args()

    if args.data_name == 'deepcrack':
        test_dataset = DeepCrackDataset(args, data_part='test')
        test_dataloaders = DataLoader(test_dataset, batch_size=8, num_workers=5)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eval(args, test_dataloaders)

   # python eval.py --data_dir "DeepCrack" --model_name LMM_Net --run_num 1 --data_name deepcrack