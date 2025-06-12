import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import glob

from model import UNet_FCN, LMM_Net
from efficientnet import EfficientCrackNet

import torchvision.transforms as transforms
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score

# 路径配置
img_dir = r'C:\Users\gb\Desktop\jiqixuexi\2\Crack-Segmentation-main\test\test image'
gt_dir = r'C:\Users\gb\Desktop\jiqixuexi\2\Crack-Segmentation-main\test\test mask'
model_paths = {
    'EfficientCrackNet': r'C:\Users\gb\Desktop\jiqixuexi\2\Crack-Segmentation-main\test\test model\EfficientCrackNet\best_model_num_3.pt',
    'LMM_Net': r'C:\Users\gb\Desktop\jiqixuexi\2\Crack-Segmentation-main\test\test model\LMM_Net\best_model_num_1.pt',
    'UNet': r'C:\Users\gb\Desktop\jiqixuexi\2\Crack-Segmentation-main\test\test model\UNet\best_model_num_1.pt'
}

# 每个模型的最佳后处理参数
model_post_params = {
    'EfficientCrackNet': {'threshold': 0.5, 'min_area': 50, 'morph_kernel': 1},
    'LMM_Net': {'threshold': 0.2, 'min_area': 1, 'morph_kernel': 1},
    'UNet': {'threshold': 0.2, 'min_area': 1, 'morph_kernel': 1}
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_TRANSFORM = {
    'UNet': (256, 256),
    'EfficientCrackNet': (192, 256),
    'LMM_Net': (112, 224)
}
def get_transform(model_name):
    size = MODEL_TRANSFORM[model_name]
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_model(model_name, weight_path):
    if model_name == 'UNet':
        model = UNet_FCN(args=None, scaler=2)
    elif model_name == 'LMM_Net':
        model = LMM_Net()
    elif model_name == 'EfficientCrackNet':
        model = EfficientCrackNet()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    checkpoint = torch.load(weight_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model

def post_process_mask(mask, threshold=0.5, min_area=10, morph_kernel=1):
    mask = (mask > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    new_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            new_mask[labels == i] = 1
    return new_mask

def evaluate_mask(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    f1 = f1_score(gt, pred)
    iou = jaccard_score(gt, pred)
    precision = precision_score(gt, pred)
    recall = recall_score(gt, pred)
    return f1, iou, precision, recall

if __name__ == '__main__':
    # 随机选10张图片
    img_list = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(img_list) < 10:
        print("图片数量不足10张，全部显示。")
        selected_imgs = img_list
    else:
        selected_imgs = random.sample(img_list, 10)

    for img_name in selected_imgs:
        img_path = os.path.join(img_dir, img_name)
        print(f"Selected image: {img_path}")

        # 读取原图
        image = Image.open(img_path).convert('RGB')
        original_size = image.size

        # 读取GT mask（支持不同扩展名）
        mask_name_wo_ext = os.path.splitext(img_name)[0]
        mask_candidates = glob.glob(os.path.join(gt_dir, mask_name_wo_ext + '.*'))
        if mask_candidates:
            mask_path = mask_candidates[0]
            gt_mask = np.array(Image.open(mask_path).convert('L'))
            gt_mask = (gt_mask > 127).astype(np.uint8)
        else:
            gt_mask = None

        # 预测并收集结果
        results = {}
        for model_name, weight_path in model_paths.items():
            model = load_model(model_name, weight_path)
            transform = get_transform(model_name)
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                output = torch.sigmoid(output)
                output_np = output.cpu().numpy().squeeze()
                # 用每个模型自己的最佳参数
                params = model_post_params[model_name]
                pred_mask = post_process_mask(
                    output_np,
                    threshold=params['threshold'],
                    min_area=params['min_area'],
                    morph_kernel=params['morph_kernel']
                )
            # 评估
            if gt_mask is not None:
                # 保证 pred_mask 和 gt_mask 尺寸一致
                if pred_mask.shape != gt_mask.shape:
                    pred_mask = np.array(Image.fromarray((pred_mask * 255).astype(np.uint8)).resize(gt_mask.shape[::-1], resample=Image.NEAREST)) // 255
                f1, iou, precision, recall = evaluate_mask(pred_mask, gt_mask)
                metrics = f"F1: {f1:.3f}\nIoU: {iou:.3f}\nPrec: {precision:.3f}\nRecall: {recall:.3f}"
            else:
                metrics = "No GT"
            results[model_name] = (pred_mask, metrics)

        # 可视化
        plt.figure(figsize=(14, 10))
        target_size = image.size  # (width, height)
        plt.subplot(2, 2, 1)
        plt.imshow(image.resize(target_size))
        plt.title('Original Image')
        plt.axis('off')

        for idx, (model_name, (mask, metrics)) in enumerate(results.items()):
            plt.subplot(2, 2, idx+2)
            mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(target_size, resample=Image.NEAREST)
            plt.imshow(mask_resized, cmap='gray')
            plt.title(model_name)
            plt.axis('off')
            plt.gca().text(0.5, -0.15, metrics, fontsize=12, ha='center', va='top', transform=plt.gca().transAxes)

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show() 