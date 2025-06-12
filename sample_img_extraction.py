import torch
import os
import cv2
import argparse
from utils import DeepCrackDataset
from torch.utils.data import DataLoader
from model import UNet_FCN, LMM_Net
from efficientnet import EfficientCrackNet
import shutil


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series forecasting of device data')
    parser.add_argument('--data_dir', type=str, help='Main directory of input dataset')
    parser.add_argument('--run_num', type=str, help='run number')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unet = UNet_FCN(scaler=2, batchnorm=True).to(device=device)
    lmmnet = LMM_Net().to(device)
    effnet = EfficientCrackNet().to(device)

    test_dataset = DeepCrackDataset(args, data_part='test')
    test_dataloaders = DataLoader(test_dataset, batch_size=8, num_workers=0)
    sample_test_img, sample_test_mask, sample_img_dir, sample_msk_dir = next(iter(test_dataloaders))

    unet.load_state_dict(torch.load(f'./saved_models/UNet/best_model_num_{args.run_num}.pt')['model_state_dict'])
    unet.eval()

    lmmnet.load_state_dict(torch.load(f'./saved_models/LMM_Net/best_model_num_{args.run_num}.pt')['model_state_dict'])
    lmmnet.eval()

    effnet.load_state_dict(torch.load(f'./saved_models/EfficientCrackNet/best_model_num_{args.run_num}.pt')['model_state_dict'])
    effnet.eval()

    for i in range(sample_test_img.shape[0]):
        with torch.no_grad():
            # Add batch to GPU
            unet_output_mask= unet(sample_test_img[i,:,:,:].unsqueeze(0).to(device))
            lmm_output_mask= lmmnet(sample_test_img[i,:,:,:].unsqueeze(0).to(device))
            effnet_output_mask= effnet(sample_test_img[i,:,:,:].unsqueeze(0).to(device))
            true_mask = sample_test_mask[i,:,:,:].unsqueeze(0)

            unet_output_mask_ = unet_output_mask.squeeze(0).permute(1, 2, 0).cpu().numpy()
            lmm_output_mask_ = lmm_output_mask.squeeze(0).permute(1, 2, 0).cpu().numpy()
            effnet_output_mask_ = effnet_output_mask.squeeze(0).permute(1, 2, 0).cpu().numpy()

            unet_output_mask_[unet_output_mask_ > 0.5] = 1
            unet_output_mask_[unet_output_mask_ < 0.5] = 0

            lmm_output_mask_[lmm_output_mask_ > 0.5] = 1
            lmm_output_mask_[lmm_output_mask_ < 0.5] = 0

            effnet_output_mask_[effnet_output_mask_ > 0.5] = 1
            effnet_output_mask_[effnet_output_mask_ < 0.5] = 0

            unet_output_mask_ = unet_output_mask_ * 255
            lmm_output_mask_ = lmm_output_mask_ * 255
            effnet_output_mask_ = effnet_output_mask_ * 255

            true_mask_ = true_mask.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img_dir, mask_dir = sample_img_dir[i], sample_msk_dir[i]
            image_name = img_dir.split("/")[-1].split("\\")[-1]
            mask_name = mask_dir.split("/")[-1].split("\\")[-1]

            shutil.copy(img_dir, f'./sample_output_images/original_images/')
            shutil.copy(mask_dir, f'./sample_output_images/ground_truths/')
            cv2.imwrite(f'./sample_output_images/UNet/{mask_name}', unet_output_mask_)
            cv2.imwrite(f'./sample_output_images/LMM_Net/{mask_name}', lmm_output_mask_)
            cv2.imwrite(f'./sample_output_images/EfficientCrackNet/{mask_name}', effnet_output_mask_)
