import torch
import os
import numpy as np
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from pathlib import Path
import glob
import random
import tifffile
from PIL import ImageFilter


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None, fused=None):

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.fused = fused
        # gives list of entire path to each image along the img_dir
        self.image_path_list = sorted(glob.glob(os.path.join(self.img_dir, "*.png"))) if not self.fused else sorted(glob.glob(os.path.join(self.img_dir, "*.tif")))
        self.mask_path_list = sorted(glob.glob(os.path.join(self.mask_dir, "*.bmp")))

        self.img_dir = img_dir  # directory for train, valid, or test 
        self.transform = transform
        self.mask_transform = mask_transform
        
    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        # TODO
        image = np.array(Image.open(self.image_path_list[idx])) if not self.fused else tifffile.imread(self.image_path_list[idx])
        image = torch.Tensor(np.moveaxis(image, [0,1,2], [2,1,0]))

        mask = np.array(Image.open(self.mask_path_list[idx]))
        mask = torch.LongTensor(np.where(mask == True, 1, 0))
        # label = self.label_idxs[idx]
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image, mask
    

class DeepCrackDataset(Dataset):
    def __init__(self, args, data_part=None):
        self.data_part = data_part
        self.augmentation_prob = 0.7  # 增加数据增强概率
        self.args = args
        
        # 初始化路径列表
        self.image_path_list = []
        self.mask_path_list = []
        
        # 根据数据部分设置路径
        if self.data_part == 'train':
            self.image_path_list = sorted(glob.glob(os.path.join(self.args.data_dir, "train_img/*.jpg"))) 
            self.mask_path_list = sorted(glob.glob(os.path.join(self.args.data_dir, "train_lab/*.png")))
        elif self.data_part == 'valid':
            self.image_path_list = sorted(glob.glob(os.path.join(self.args.data_dir, "valid_img/*.jpg"))) 
            self.mask_path_list = sorted(glob.glob(os.path.join(self.args.data_dir, "valid_lab/*.png")))
        elif self.data_part == 'test':
            self.image_path_list = sorted(glob.glob(os.path.join(self.args.data_dir, "test_img/*.jpg"))) 
            self.mask_path_list = sorted(glob.glob(os.path.join(self.args.data_dir, "test_lab/*.png")))
        
        if not self.image_path_list or not self.mask_path_list:
            raise ValueError(f"在 {self.args.data_dir} 中找不到 {self.data_part} 数据集的图像或掩码文件")
            
        print(f"加载 {self.data_part} 数据集: {len(self.image_path_list)} 个图像")
        
    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        # 加载图像和掩码
        image = Image.open(self.image_path_list[idx])
        mask = Image.open(self.mask_path_list[idx])

        # 基础变换
        transformer = []
        mask_transformer = []
        if self.args.model_name == 'LMM_Net':
            image_width, image_height = 112, 224
            mask_width, mask_height = 112, 224
        elif self.args.model_name == 'EfficientCrackNet':
            image_width, image_height = 192, 256
            mask_width, mask_height = 192, 256
        else:  # UNet
            image_width, image_height = 256, 256
            mask_width, mask_height = 256, 256

        transformer.append(transforms.Resize((image_width, image_height)))
        mask_transformer.append(transforms.Resize((mask_width, mask_height)))

        # 训练集数据增强
        if self.data_part == 'train':
            # 随机旋转
            if random.random() < self.augmentation_prob:
                angle = random.uniform(-15, 15)
                image = F.rotate(image, angle)
                mask = F.rotate(mask, angle)

            # 随机翻转
            if random.random() < self.augmentation_prob:
                image = F.hflip(image)
                mask = F.hflip(mask)
            if random.random() < self.augmentation_prob:
                image = F.vflip(image)
                mask = F.vflip(mask)
                
            # 随机裁剪和缩放
            if random.random() < self.augmentation_prob:
                scale = random.uniform(0.8, 1.2)
                new_size = (int(image_width * scale), int(image_height * scale))
                image = F.resize(image, new_size)
                mask = F.resize(mask, new_size)
                # 只有当新尺寸大于等于目标尺寸时才进行随机裁剪，否则直接resize回目标尺寸
                if new_size[0] >= image_width and new_size[1] >= image_height:
                    i = random.randint(0, new_size[0] - image_width)
                    j = random.randint(0, new_size[1] - image_height)
                    image = F.crop(image, i, j, image_width, image_height)
                    mask = F.crop(mask, i, j, mask_width, mask_height)
                else:
                    image = F.resize(image, (image_width, image_height))
                    mask = F.resize(mask, (mask_width, mask_height))

            # 颜色增强
            if random.random() < self.augmentation_prob:
                # 亮度、对比度、饱和度、色调调整
                image = F.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = F.adjust_contrast(image, random.uniform(0.8, 1.2))
                image = F.adjust_saturation(image, random.uniform(0.8, 1.2))
                image = F.adjust_hue(image, random.uniform(-0.1, 0.1))

            # 随机噪声
            if random.random() < self.augmentation_prob:
                # 将图像转换为float类型并归一化到[0,1]范围
                image_np = np.array(image).astype(np.float32) / 255.0
                # 生成噪声
                noise = np.random.normal(0, 0.1, image_np.shape).astype(np.float32)
                # 添加噪声并裁剪到[0,1]范围
                image_np = np.clip(image_np + noise, 0, 1)
                # 转回PIL图像
                image = Image.fromarray((image_np * 255).astype(np.uint8))

            # 随机模糊
            if random.random() < self.augmentation_prob:
                blur_kernel = random.choice([3, 5])
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.5)))

        # 转换为张量
        transformer.append(transforms.ToTensor())
        transformer = transforms.Compose(transformer)

        image = transformer(image)
        mask = transformer(mask)

        if mask.shape[0] > 1:
            transformer = transforms.Grayscale(num_output_channels=1)
            mask = transformer(mask)

        mask[mask < 0.5] = 0
        mask[mask > 0.5] = 1

        return image, mask
    

def save_training_plot_only(epoch_train_loss, epochs, args):
    plt.figure(figsize=(14, 5))

  # Accuracy plot
    plt.subplot(1, 2, 1)
    train_loss_plot, = plt.plot(epochs, epoch_train_loss, 'r')
    plt.title('Training Loss')
    plt.legend([train_loss_plot], ['Training Loss'])
    plt.savefig(f'./plots/{args.model_name}/run_{args.run_num}/loss_plots.jpg')


def save_plots(epoch_train_loss, epoch_valid_loss, epochs, args):
    plt.figure(figsize=(14, 5))

  # Accuracy plot
    plt.subplot(1, 2, 1)
    train_loss_plot, = plt.plot(epochs, epoch_train_loss, 'r')
    val_loss_plot, = plt.plot(epochs, epoch_valid_loss, 'b')
    plt.title('Training and Validation Loss')
    plt.legend([train_loss_plot, val_loss_plot], ['Training Loss', 'Validation Loss'])
    plt.savefig(f'./plots/{args.model_name}/run_{args.run_num}/loss_plots.jpg')


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

    
def save_checkpoint(save_path, model, loss, val_used=False):
    """保存模型权重和损失值"""
    if save_path is not None:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存模型状态字典和损失值
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_loss' if val_used else 'train_loss': loss
        }, save_path)
        print(f"模型已保存到: {save_path}")