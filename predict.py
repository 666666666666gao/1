import torch
from efficientnet import EfficientCrackNet
from model import UNet_FCN, LMM_Net
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import itertools
from sklearn.metrics import f1_score, jaccard_score
from utils import CustomImageDataset, DeepCrackDataset
from torch.utils.data import DataLoader
import warnings
import json
warnings.filterwarnings("ignore")

# 统一transform函数
MODEL_TRANSFORM = {
    'UNet': (256, 256),
    'EfficientCrackNet': (192, 256),
    'LMM_Net': (112, 224)
}

def get_transform(model_name):
    if model_name not in MODEL_TRANSFORM:
        raise ValueError(f"不支持的模型名称: {model_name}")
    size = MODEL_TRANSFORM[model_name]
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        # 归一化参数可根据训练时实际情况调整
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def post_process_mask(mask, min_area=100, morph_kernel=3):
    # mask: torch tensor, shape (1, H, W) or (H, W)
    mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)
    # 形态学开运算去小噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
    # 形态学闭运算填补小空洞
    mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
    # 去除小连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    new_mask = np.zeros_like(mask_np)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            new_mask[labels == i] = 1
    return new_mask

def get_model(args):
    """根据模型名称返回对应的模型实例"""
    if args.model_name == 'UNet':
        model = UNet_FCN(args=args, scaler=2)
    elif args.model_name == 'LMM_Net':
        model = LMM_Net()
    elif args.model_name == 'EfficientCrackNet':
        model = EfficientCrackNet()
    else:
        raise ValueError(f"不支持的模型名称: {args.model_name}")
    return model

def evaluate_mask(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    f1 = f1_score(gt, pred)
    iou = jaccard_score(gt, pred)
    return f1, iou

def save_best_params(model_name, best_params, best_score, save_dir='best_params'):
    """保存最佳参数到JSON文件"""
    os.makedirs(save_dir, exist_ok=True)
    params_dict = {
        'model_name': model_name,
        'threshold': float(best_params[0]),
        'min_area': int(best_params[1]),
        'morph_kernel': int(best_params[2]),
        'best_f1_score': float(best_score)
    }
    save_path = os.path.join(save_dir, f'{model_name}_best_params.json')
    with open(save_path, 'w') as f:
        json.dump(params_dict, f, indent=4)
    print(f"最佳参数已保存到: {save_path}")
    return save_path

def load_best_params(model_name, params_dir='best_params'):
    """从JSON文件加载最佳参数"""
    params_path = os.path.join(params_dir, f'{model_name}_best_params.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
        print(f"已加载最佳参数: {params}")
        return params
    return None

def auto_search_params(model, image_files, mask_files, device, transform, param_grid, input_dir, mask_dir, model_name):
    best_score = -1
    best_params = None
    total_combinations = len(param_grid['threshold']) * len(param_grid['min_area']) * len(param_grid['morph_kernel'])
    print(f"开始搜索最佳参数，共 {total_combinations} 种组合...")
    
    for threshold, min_area, morph_kernel in itertools.product(param_grid['threshold'], param_grid['min_area'], param_grid['morph_kernel']):
        f1s, ious = [], []
        for img_name, mask_name in zip(image_files, mask_files):
            # 构建完整的文件路径
            img_path = os.path.join(input_dir, img_name)
            mask_path = os.path.join(mask_dir, mask_name)
            
            if not os.path.exists(img_path):
                print(f"警告: 图片文件不存在: {img_path}")
                continue
            if not os.path.exists(mask_path):
                print(f"警告: mask文件不存在: {mask_path}")
                continue
                
            # 读取原始图片和mask
            image = Image.open(img_path).convert('RGB')
            gt_mask = np.array(Image.open(mask_path).convert('L'))
            gt_mask = (gt_mask > 127).astype(np.uint8)
            
            # 保存原始尺寸
            original_size = gt_mask.shape
            
            # 模型推理
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                pred_mask = (output > threshold).float()
                pred_mask_np = post_process_mask(pred_mask, min_area=min_area, morph_kernel=morph_kernel)
            
            # 将预测mask调整到原始尺寸
            pred_mask_np = cv2.resize(pred_mask_np, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
            
            # 评估
            f1, iou = evaluate_mask(pred_mask_np, gt_mask)
            f1s.append(f1)
            ious.append(iou)
        mean_f1 = np.mean(f1s)
        mean_iou = np.mean(ious)
        print(f"Params: threshold={threshold}, min_area={min_area}, morph_kernel={morph_kernel} | F1={mean_f1:.4f}, IoU={mean_iou:.4f}")
        if mean_f1 > best_score:
            best_score = mean_f1
            best_params = (threshold, min_area, morph_kernel)
            # 保存当前最佳参数
            save_best_params(model_name, best_params, best_score)
    
    print(f"最佳参数: threshold={best_params[0]}, min_area={best_params[1]}, morph_kernel={best_params[2]} | F1={best_score:.4f}")
    return best_params

def predict_image(model, image_path, device, output_dir, transform, threshold=0.5, min_area=100, morph_kernel=3, save=True):
    # 预处理图片
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # 保存原始尺寸
    image_resized = image.resize((256, 192))  # 仅用于可视化
    input_tensor = transform(image).unsqueeze(0).to(device)  # 增加batch维度

    # 推理
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = (output > threshold).float()
        # 后处理
        pred_mask_np = post_process_mask(pred_mask, min_area=min_area, morph_kernel=morph_kernel)
        
    # 将预测mask调整到原始尺寸
    pred_mask_np = cv2.resize(pred_mask_np, (original_size[0], original_size[1]), interpolation=cv2.INTER_NEAREST)

    # 保存结果
    if save:
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.title('Input Image')
        plt.imshow(image_resized)
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.title('Predicted Mask')
        plt.imshow(pred_mask_np, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        
        # 创建输出文件名
        output_path = os.path.join(output_dir, f"{Path(image_path).stem}_prediction.png")
        plt.savefig(output_path)
        plt.close()

    return pred_mask_np

def predict(args):
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'使用设备: {device}')

    # 创建保存目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'预测结果将保存到: {args.output_dir}')

    # 加载模型
    model = get_model(args)
    model = model.to(device)
    
    # 加载权重
    if not args.model_path:
        args.model_path = f'./saved_models/{args.model_name}/best_model_num_{args.run_num}.pt'
    
    print(f'正在从 {args.model_path} 加载模型权重...')
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功加载模型权重，验证损失: {checkpoint.get('val_loss', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
        print("成功加载模型权重")
    model.eval()

    # 获取输入图片列表
    image_files = [f for f in os.listdir(args.input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise ValueError(f"在 {args.input_dir} 中没有找到图片文件")

    print(f'找到 {len(image_files)} 张图片')

    # 如果启用了自动搜索参数
    if args.auto_search and args.mask_dir:
        print("开始自动搜索最佳参数...")
        # 首先尝试加载已保存的最佳参数
        saved_params = load_best_params(args.model_name)
        if saved_params and not args.force_search:
            print("使用已保存的最佳参数...")
            args.threshold = saved_params['threshold']
            args.min_area = saved_params['min_area']
            args.morph_kernel = saved_params['morph_kernel']
        else:
            print("开始新的参数搜索...")
            param_grid = {
                'threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                'min_area': [1, 10, 30, 50, 100, 120, 150],
                'morph_kernel': [1, 3, 5, 7]
            }
            mask_files = [f for f in os.listdir(args.mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            best_params = auto_search_params(model, image_files, mask_files, device, get_transform(args.model_name), 
                                           param_grid, args.input_dir, args.mask_dir, args.model_name)
            args.threshold, args.min_area, args.morph_kernel = best_params
            print(f"使用最佳参数: threshold={args.threshold}, min_area={args.min_area}, morph_kernel={args.morph_kernel}")

    # 预测每张图片
    print('开始预测...')
    saved_count = 0
    for img_name in tqdm(image_files):
        try:
            img_path = os.path.join(args.input_dir, img_name)
            pred_mask = predict_image(model, img_path, device, args.output_dir, 
                                    get_transform(args.model_name),
                                    threshold=args.threshold,
                                    min_area=args.min_area,
                                    morph_kernel=args.morph_kernel)
            saved_count += 1
        except Exception as e:
            print(f'处理图片 {img_name} 时出错: {str(e)}')

    print(f'预测完成! 成功保存 {saved_count} 张图片到: {args.output_dir}')
    
    # 验证保存的文件
    saved_files = os.listdir(args.output_dir)
    print(f'目录中的文件数量: {len(saved_files)}')
    if len(saved_files) > 0:
        print('前5个保存的文件:')
        for f in saved_files[:5]:
            print(f'- {f}')

def main():
    parser = argparse.ArgumentParser(description='Crack Segmentation Prediction')
    parser.add_argument('--model_name', type=str, required=True, help='模型名称 (UNet, LMM_Net, EfficientCrackNet)')
    parser.add_argument('--model_path', type=str, help='模型权重文件路径')
    parser.add_argument('--input_dir', type=str, default='predict', help='输入图片目录')
    parser.add_argument('--output_dir', type=str, default='predictions', help='输出预测结果目录')
    parser.add_argument('--threshold', type=float, default=0.5, help='预测阈值')
    parser.add_argument('--min_area', type=int, default=100, help='最小连通区域面积')
    parser.add_argument('--morph_kernel', type=int, default=3, help='形态学操作核大小')
    parser.add_argument('--batch_size', type=int, default=8, help='批量大小')
    parser.add_argument('--data_name', type=str, default='deepcrack', help='数据集名称')
    parser.add_argument('--rgb', type=bool, default=False, help='是否为RGB图像')
    parser.add_argument('--run_num', type=str, help='运行编号')
    parser.add_argument('--auto_search', action='store_true', help='是否自动搜索最佳参数')
    parser.add_argument('--mask_dir', type=str, help='用于自动搜索参数的mask目录')
    parser.add_argument('--force_search', action='store_true', help='强制重新搜索参数，忽略已保存的参数')
    args = parser.parse_args()

    predict(args)

if __name__ == '__main__':
    main()
    #指定参数 python predict.py --model_name UNet --model_path "saved_models/UNet/best_model_num_1.pt" --input_dir "predict" --output_dir "predictions/UNet" --threshold 0.4 --min_area 120 --morph_kernel 5
    #自动搜索参数 python predict.py --model_name UNet --model_path "saved_models/UNet/best_model_num_1.pt" --input_dir "DeepCrack/test_img" --mask_dir "DeepCrack/test_lab" --auto_search --output_dir predictions/UNet
    #efficiencycracknet 最佳参数 threshold=0.5, min_area=50, morph_kernel=1 （best_model_num_3）
    #UNet 最佳参数 threshold=0.4, min_area=120, morph_kernel=5 （best_model_num_1）

    #新增功能说明：
    #1. 参数保存功能：
    #   - 自动搜索到的最佳参数会保存在 best_params 目录下
    #   - 文件名格式：{model_name}_best_params.json
    #   - 保存内容：threshold, min_area, morph_kernel 和对应的 F1 分数
    #   示例：python predict.py --model_name UNet --auto_search --mask_dir "DeepCrack/test_lab" --input_dir "DeepCrack/test_img"

    #2. 参数加载功能：
    #   - 如果已经保存过参数，下次运行时会自动加载已保存的参数
    #   - 无需重新搜索，直接使用历史最佳参数
    #   示例：python predict.py --model_name UNet --auto_search --mask_dir "DeepCrack/test_lab" --input_dir "DeepCrack/test_img"

    #3. 强制重新搜索：
    #   - 使用 --force_search 参数可以强制重新搜索，忽略已保存的参数
    #   - 适用于想要重新评估参数或数据集发生变化的情况
    #   示例：python predict.py --model_name UNet --auto_search --mask_dir "DeepCrack/test_lab" --input_dir "DeepCrack/test_img" --force_search
