import torch
import os 
import pickle
import gc
import argparse
import numpy as np
import torch.nn as nn
from torch import optim
from utils import CustomImageDataset, DeepCrackDataset, save_plots, save_training_plot_only, save_checkpoint, init_weights
from model import UNet_FCN, LMM_Net
from efficientnet import EfficientCrackNet
from loss_functions import BCELoss, DiceLoss, IoULoss
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Subset
import warnings
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, jaccard_score
import matplotlib.pyplot as plt
print(torch.cuda.is_available())
warnings.filterwarnings("ignore")

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, verbose=True, mode='min', monitor='val_loss'):
        """
        增强版早停机制
        Args:
            patience: 容忍多少个epoch没有改善
            min_delta: 最小改善阈值
            verbose: 是否打印详细信息
            mode: 'min' 或 'max'，表示监控指标是越小越好还是越大越好
            monitor: 监控的指标名称，可以是 'val_loss', 'f1_score', 'iou' 等
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.mode = mode
        self.monitor = monitor
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        # 保存最佳模型状态
        self.best_model_state = None
        self.best_optimizer_state = None
        
        # 用于存储训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'f1_score': [],
            'iou': [],
            'learning_rate': []
        }
        
        # 学习率调整相关
        self.lr_patience = 3
        self.lr_counter = 0
        self.min_lr = 1e-6
        
    def __call__(self, metrics, model, optimizer, epoch, path):
        """
        检查是否需要早停
        Args:
            metrics: 包含各种指标的字典
            model: 模型实例
            optimizer: 优化器实例
            epoch: 当前epoch
            path: 模型保存路径
        """
        current_score = metrics[self.monitor]
        
        if self.best_score is None:
            self.best_score = current_score
            self._save_checkpoint(metrics, model, optimizer, epoch, path)
        elif (self.mode == 'min' and current_score > self.best_score - self.min_delta) or \
             (self.mode == 'max' and current_score < self.best_score + self.min_delta):
            self.counter += 1
            self.lr_counter += 1
            
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                print(f'Current {self.monitor}: {current_score:.4f}, Best {self.monitor}: {self.best_score:.4f}')
            
            # 检查是否需要降低学习率
            if self.lr_counter >= self.lr_patience:
                self._adjust_learning_rate(optimizer)
                self.lr_counter = 0
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'Early stopping triggered. Restoring best model from epoch {self.best_epoch}')
                self._restore_best_model(model, optimizer)
        else:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            self.lr_counter = 0
            self._save_checkpoint(metrics, model, optimizer, epoch, path)
            if self.verbose:
                print(f'Validation {self.monitor} improved ({self.best_score:.4f}). Saving model ...')
        
        # 更新历史记录
        for key in metrics:
            if key in self.history:
                self.history[key].append(metrics[key])
        self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])
    
    def _save_checkpoint(self, metrics, model, optimizer, epoch, path):
        """保存检查点"""
        self.best_model_state = model.state_dict().copy()
        self.best_optimizer_state = optimizer.state_dict().copy()
        
        # 保存完整的检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }, path)
    
    def _restore_best_model(self, model, optimizer):
        """恢复最佳模型状态"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
        if self.best_optimizer_state is not None:
            optimizer.load_state_dict(self.best_optimizer_state)
    
    def _adjust_learning_rate(self, optimizer):
        """调整学习率"""
        current_lr = optimizer.param_groups[0]['lr']
        new_lr = max(current_lr * 0.5, self.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        if self.verbose:
            print(f'Reducing learning rate to {new_lr:.6f}')
    
    def plot_history(self, save_path):
        """绘制训练历史"""
        plt.figure(figsize=(15, 10))
        
        # 绘制损失
        plt.subplot(2, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 绘制F1分数
        plt.subplot(2, 2, 2)
        plt.plot(self.history['f1_score'], label='F1 Score')
        plt.title('F1 Score History')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        
        # 绘制IoU
        plt.subplot(2, 2, 3)
        plt.plot(self.history['iou'], label='IoU')
        plt.title('IoU History')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()
        
        # 绘制学习率
        plt.subplot(2, 2, 4)
        plt.plot(self.history['learning_rate'], label='Learning Rate')
        plt.title('Learning Rate History')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def get_model(args):
    """根据模型名称返回对应的模型实例"""
    if args.model_name == 'UNet':
        model = UNet_FCN(args=args, scaler=2)
        model.apply(init_weights)
    elif args.model_name == 'LMM_Net':
        model = LMM_Net()
    elif args.model_name == 'EfficientCrackNet':
        model = EfficientCrackNet()
    else:
        raise ValueError(f"不支持的模型名称: {args.model_name}")
    return model

def get_criterion(args):
    """返回组合损失函数"""
    bce_loss = BCELoss()
    dice_loss = DiceLoss()
    miou_loss = IoULoss()
    
    def combined_loss(outputs, targets):
        bce = bce_loss(outputs, targets)
        dice = dice_loss(outputs, targets)
        iou = miou_loss(outputs, targets)
        return bce + (args.alpha * (dice + iou))
    
    return combined_loss

def objective(trial, args, full_train_dataset):
    # 定义超参数搜索空间
    args.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    args.batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
    args.alpha = trial.suggest_float('alpha', 0.1, 1.0)
    args.optim_w_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    # 使用K折交叉验证
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_train_dataset)):
        # 创建数据加载器
        train_subsampler = Subset(full_train_dataset, train_ids)
        val_subsampler = Subset(full_train_dataset, val_ids)
        
        train_loader = DataLoader(train_subsampler, batch_size=args.batch_size, num_workers=10)
        val_loader = DataLoader(val_subsampler, batch_size=args.batch_size, num_workers=10)
        
        # 初始化模型
        if args.model_name == 'UNet':
            model = UNet_FCN(args=args, scaler=2).to(device)
            model.apply(init_weights)
        elif args.model_name == 'LMM_Net':
            model = LMM_Net().to(device)
        elif args.model_name == 'EfficientCrackNet':
            model = EfficientCrackNet().to(device)
            
        # 加载预训练权重
        if args.pretrained and os.path.exists(args.pretrained):
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # 定义损失函数和优化器
        bce_loss = BCELoss()
        dice_loss = DiceLoss()
        miou_loss = IoULoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.optim_w_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # 训练一个epoch
        model.train()
        train_loss = 0
        for inputs, masks in train_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = bce_loss(outputs, masks) + (args.alpha * (dice_loss(outputs, masks) + miou_loss(outputs, masks)))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, masks in val_loader:
                inputs, masks = inputs.to(device), masks.to(device)
                outputs = model(inputs)
                loss = bce_loss(outputs, masks)
                val_loss += loss.item()
        
        cv_scores.append(val_loss / len(val_loader))
    
    return np.mean(cv_scores)


def optimize_hyperparameters(args, train_dataset):
    # 设置环境变量以优化CUDA内存分配
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device  # 添加device属性到args
    
    # 减小默认批量大小
    args.batch_size = 4  # 从原来的8或16减小到4
    
    # 减小图像尺寸
    args.img_size = (96, 192)  # 从原来的(112, 224)减小到(96, 192)
    
    def objective(trial, args, dataset):
        # 设置超参数搜索范围
        args.learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
        args.weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        args.alpha = trial.suggest_float('alpha', 0.1, 1.0)  # 损失函数权重
        args.batch_size = trial.suggest_categorical('batch_size', [2, 4, 8, 16])
        
        # 优化器相关参数
        args.optimizer = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
        if args.optimizer == 'sgd':
            args.momentum = trial.suggest_float('momentum', 0.8, 0.99)
            args.nesterov = trial.suggest_categorical('nesterov', [True, False])
        
        # 学习率调度器参数
        args.lr_scheduler = trial.suggest_categorical('lr_scheduler', ['cosine', 'step', 'plateau'])
        if args.lr_scheduler == 'step':
            args.lr_step_size = trial.suggest_int('lr_step_size', 5, 20)
            args.lr_gamma = trial.suggest_float('lr_gamma', 0.1, 0.9)
        elif args.lr_scheduler == 'plateau':
            args.lr_patience = trial.suggest_int('lr_patience', 3, 10)
            args.lr_factor = trial.suggest_float('lr_factor', 0.1, 0.5)
        
        # 数据增强参数
        args.augmentation_prob = trial.suggest_float('augmentation_prob', 0.5, 0.9)
        
        # 创建数据加载器
        train_loader = DataLoader(dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=2, pin_memory=True)
        
        # 初始化模型
        model = get_model(args).to(device)
        
        # 选择优化器
        if args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, 
                                 weight_decay=args.weight_decay)
        elif args.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, 
                                  weight_decay=args.weight_decay)
        else:  # sgd
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, 
                                momentum=args.momentum, nesterov=args.nesterov,
                                weight_decay=args.weight_decay)
        
        # 选择学习率调度器
        if args.lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
        elif args.lr_scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
            )
        else:  # plateau
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=args.lr_factor, 
                patience=args.lr_patience, verbose=True
            )
        
        criterion = get_criterion(args)
        
        # 使用梯度累积
        accumulation_steps = 4  # 累积4步再更新
        optimizer.zero_grad()
        
        model.train()
        total_loss = 0
        try:
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 缩放损失以适应梯度累积
                loss = loss / accumulation_steps
                loss.backward()
                
                # 梯度累积
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # 更新学习率
                    if args.lr_scheduler == 'plateau':
                        scheduler.step(loss)
                    else:
                        scheduler.step()
                
                total_loss += loss.item() * accumulation_steps
                
                # 清理缓存
                if i % 10 == 0:  # 每10个batch清理一次
                    torch.cuda.empty_cache()
            
            return total_loss / len(train_loader)
        except Exception as e:
            print(f"Trial failed with error: {str(e)}")
            return float('inf')  # 返回无穷大表示这个trial失败
    
    print("\n开始超参数优化...")
    print(f"使用设备: {device}")
    print(f"批量大小: {args.batch_size}")
    print(f"图像尺寸: {args.img_size}")
    
    # 创建study对象
    study = optuna.create_study(direction='minimize')
    
    # 增加搜索次数
    study.optimize(lambda trial: objective(trial, args, train_dataset), n_trials=20)
    
    # 更新最佳参数
    best_params = study.best_params
    args.learning_rate = best_params['learning_rate']
    args.weight_decay = best_params['weight_decay']
    args.alpha = best_params['alpha']
    args.batch_size = best_params['batch_size']
    args.optimizer = best_params['optimizer']
    args.lr_scheduler = best_params['lr_scheduler']
    args.augmentation_prob = best_params['augmentation_prob']
    
    # 根据优化器类型更新相关参数
    if args.optimizer == 'sgd':
        args.momentum = best_params['momentum']
        args.nesterov = best_params['nesterov']
    
    # 根据学习率调度器类型更新相关参数
    if args.lr_scheduler == 'step':
        args.lr_step_size = best_params['lr_step_size']
        args.lr_gamma = best_params['lr_gamma']
    elif args.lr_scheduler == 'plateau':
        args.lr_patience = best_params['lr_patience']
        args.lr_factor = best_params['lr_factor']
    
    print("\n最佳超参数:")
    print(f"学习率: {args.learning_rate:.6f}")
    print(f"权重衰减: {args.weight_decay:.6f}")
    print(f"Alpha: {args.alpha:.4f}")
    print(f"批量大小: {args.batch_size}")
    print(f"优化器: {args.optimizer}")
    print(f"学习率调度器: {args.lr_scheduler}")
    print(f"数据增强概率: {args.augmentation_prob:.2f}")
    
    return args


def run_with_validation(args, model, train_dataloaders, valid_dataloaders, plot_path):
    # Training loop
    epochs = []                
    epoch_train_loss = []
    epoch_valid_loss = []
    epoch_f1_scores = []
    epoch_ious = []
    
    # 初始化最佳验证损失
    best_val_loss = float('inf')
    
    # 修改保存路径，将run_num加入文件名
    best_path = f'./saved_models/{args.model_name}/best_model_num_{args.run_num}.pt'
    
    # 初始化损失函数
    bce_loss = BCELoss()
    dice_loss = DiceLoss()
    miou_loss = IoULoss()

    # 初始化早停
    early_stopping = EarlyStopping(
        patience=10,
        min_delta=0.001,
        verbose=True,
        mode='min',
        monitor='val_loss'
    )

    # 使用余弦退火学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.optim_w_decay)
    
    print('------------  训练开始! --------------')
    print(f'初始学习率: {args.learning_rate:.6f}')
    print(f'批量大小: {args.batch_size}')
    print(f'Alpha值: {args.alpha}')
    print(f'权重衰减: {args.optim_w_decay}')
    print(f'模型权重将保存到: {best_path}')
    
    num_epochs = args.epochs
    alpha = args.alpha
    min_alpha = 0.1  # 设置alpha的最小值
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        train_f1 = 0
        train_iou = 0
        
        # 训练阶段
        for i, (inputs, masks) in enumerate(train_dataloaders):
            inputs, masks = inputs.to(device), masks.to(device)
            
            optimizer.zero_grad()    
            outputs = model(inputs)

            # 计算损失
            bce = bce_loss(outputs, masks)
            dice = dice_loss(outputs, masks)
            iou = miou_loss(outputs, masks)
            loss = bce + (alpha * (dice + iou))
            
            # 计算指标
            pred_masks = (outputs > 0.5).float()
            f1 = f1_score(masks.cpu().numpy().flatten(), pred_masks.cpu().numpy().flatten())
            iou_score = jaccard_score(masks.cpu().numpy().flatten(), pred_masks.cpu().numpy().flatten())
            
            train_loss += loss.item()
            train_f1 += f1
            train_iou += iou_score

            loss.backward()
            optimizer.step()
            
            # 清理缓存
            del inputs, masks, outputs
            torch.cuda.empty_cache()
        
        # 计算平均训练指标
        train_loss /= len(train_dataloaders)
        train_f1 /= len(train_dataloaders)
        train_iou /= len(train_dataloaders)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_f1 = 0
        val_iou = 0
        
        with torch.no_grad():
            for inputs, masks in valid_dataloaders:
                inputs, masks = inputs.to(device), masks.to(device)
                outputs = model(inputs)
                
                # 计算损失
                bce = bce_loss(outputs, masks)
                dice = dice_loss(outputs, masks)
                iou = miou_loss(outputs, masks)
                loss = bce + (alpha * (dice + iou))
                
                # 计算指标
                pred_masks = (outputs > 0.5).float()
                f1 = f1_score(masks.cpu().numpy().flatten(), pred_masks.cpu().numpy().flatten())
                iou_score = jaccard_score(masks.cpu().numpy().flatten(), pred_masks.cpu().numpy().flatten())
                
                val_loss += loss.item()
                val_f1 += f1
                val_iou += iou_score
                
                # 清理缓存
                del inputs, masks, outputs
                torch.cuda.empty_cache()
        
        # 计算平均验证指标
        val_loss /= len(valid_dataloaders)
        val_f1 /= len(valid_dataloaders)
        val_iou /= len(valid_dataloaders)
        
        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印当前epoch的结果
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Learning Rate: {current_lr:.6f}')
        print(f'Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Train IoU: {train_iou:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val IoU: {val_iou:.4f}')
        
        # 更新早停
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'f1_score': val_f1,
            'iou': val_iou
        }
        early_stopping(metrics, model, optimizer, epoch, best_path)
        
        # 如果触发早停，保存训练历史并退出
        if early_stopping.early_stop:
            print("触发早停机制，停止训练")
            early_stopping.plot_history(os.path.join(plot_path, 'training_history.png'))
            break
        
        # 更新alpha值
        if epoch % 20 == 0:
            alpha = max(alpha - 0.2, min_alpha)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = f'./saved_models/{args.model_name}/best_model_num_{args.run_num}.pt'
            torch.save(model.state_dict(), best_path)
            print(f"保存最佳模型到: {best_path}")
    
    print("训练完成!")
    # 保存最终训练历史
    early_stopping.plot_history(os.path.join(plot_path, 'training_history.png'))

def run_without_validation(args, model, train_dataloaders, plot_path):
    # Training loop
    epochs = []                
    epoch_train_loss = []
    best_loss = np.inf
    # 修改保存路径，将run_num加入文件名
    best_path = f'./saved_models/{args.model_name}/best_model_num_{args.run_num}.pt'
    bce_loss = BCELoss()
    dice_loss = DiceLoss()
    miou_loss = IoULoss()

    # 使用余弦退火学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.optim_w_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # 第一次重启的周期
        T_mult=2,  # 每次重启后周期长度的倍数
        eta_min=1e-6  # 最小学习率
    )

    print('------------  训练开始! --------------')
    print(f'初始学习率: {args.learning_rate:.6f}')
    print(f'批量大小: {args.batch_size}')
    print(f'Alpha值: {args.alpha}')
    print(f'权重衰减: {args.optim_w_decay}')
    print(f'模型权重将保存到: {best_path}')

    num_epochs = args.epochs
    alpha = args.alpha
    for epoch in tqdm(range(num_epochs)):
        model.train()

        b_train_loss = []
        if epoch % 60 == 0:
            alpha -= 0.2

        for i, (inputs, labels) in enumerate(train_dataloaders):
            inputs, masks = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()    
            outputs = model(inputs)
            loss = bce_loss(outputs, masks.to(device)) + (alpha * (dice_loss(outputs, masks.to(device)) + miou_loss(outputs, masks.to(device))))
            b_train_loss.append(loss.item())

            del inputs
            del labels
            torch.cuda.empty_cache()
            gc.collect()
            
            loss.backward()
            optimizer.step()

        if np.mean(b_train_loss) < best_loss:
            best_loss = np.mean(b_train_loss)
            save_checkpoint(best_path, model, np.mean(b_train_loss), args.validate)
        
        epoch_train_loss.append(np.mean(b_train_loss))
        lr_scheduler.step()  # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch+1}, 学习率: {current_lr:.6f}')
        print(f'训练损失: {np.mean(b_train_loss):.4f}')

        epochs.append(epoch+1)

    print("训练完成!")
    save_training_plot_only(epoch_train_loss, epochs, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series forecasting of device data')
    parser.add_argument('--data_dir', type=str, help='Main directory of input dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,help='Momentum in learning rate')
    parser.add_argument('--validate', type=bool, default=False, help='whether to validate the model or not')
    parser.add_argument('--model_name', type=str, help='algorithm')
    parser.add_argument('--epochs', type=int, help='Num of epochs')
    parser.add_argument('--alpha', type=float, default=0.8, help='Reduction factor in Loss function')
    parser.add_argument('--optim_w_decay', type=float, default=2e-4)
    parser.add_argument('--lr_decay', type=float, default=0.8)
    parser.add_argument('--num_epochs_decay', type=int, default=5)
    parser.add_argument('--data_name', type=str, help='Dataset to be used')
    parser.add_argument('--rgb', type=bool, help='Is image RGB or not')
    parser.add_argument('--run_num', type=str, help='run number')
    parser.add_argument('--half', type=bool, default=False, help='use half Model size or not')
    parser.add_argument('--augment', type=bool, default=False, help='whether augment dataset or not')
    parser.add_argument('--pretrained', type=str, default=None, help='预训练模型路径')
    parser.add_argument('--optimize', type=bool, default=False, help='是否进行超参数优化')
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device  # 添加device属性到args
    
    # 创建保存目录
    save_dir = f'./saved_models/{args.model_name}'
    plot_dir = f'./plots/{args.model_name}/run_{args.run_num}'
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f'模型权重将保存到: {save_dir}')
    print(f'训练曲线将保存到: {plot_dir}')
    
    # 准备数据加载器
    if args.data_name == 'deepcrack':
        train_dataset = DeepCrackDataset(args, data_part='train')
        
        # 如果启用超参数优化
        if args.optimize:
            args = optimize_hyperparameters(args, train_dataset)
            
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=10)
        
        valid_dataloaders = None
        if args.validate:
            try:
                valid_dataset = DeepCrackDataset(args, data_part='valid')
                valid_dataloaders = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=10)
            except ValueError as e:
                print(f"警告: {e}")
                print("将使用测试集作为验证集")
                try:
                    valid_dataset = DeepCrackDataset(args, data_part='test')
                    valid_dataloaders = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=10)
                    args.validate = True
                except ValueError as e:
                    print(f"警告: {e}")
                    print("将使用无验证模式运行")
                    args.validate = False
    else:
        raise ValueError(f"不支持的数据集名称: {args.data_name}")

    # 初始化模型
    if args.model_name == 'UNet':
        model = UNet_FCN(args=args, scaler=2).to(device)
        model.apply(init_weights)
    elif args.model_name == 'LMM_Net':
        model = LMM_Net().to(device)
    elif args.model_name == 'EfficientCrackNet':
        model = EfficientCrackNet().to(device)
    else:
        raise ValueError(f"不支持的模型名称: {args.model_name}")

    # 加载预训练权重
    if args.pretrained and os.path.exists(args.pretrained):
        print(f"加载预训练权重: {args.pretrained}")
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("预训练权重加载成功!")
    elif args.pretrained:
        print(f"警告: 预训练权重文件 {args.pretrained} 不存在，将使用随机初始化的权重")

    print('CUDA是否可用: ', torch.cuda.is_available())

    # 始终使用带验证的模式运行，如果没有验证集则使用测试集
    run_with_validation(args, model, train_dataloaders, valid_dataloaders, plot_dir)

# python main_dev.py --data_dir C:/Users/jwkor/Documents/UNM/crack_segmentation/dataset/DeepCrack/ --validate True --model_name UNet --epochs 50 --alpha 0.8 --data_name deepcrack --run_num 1
# python main_dev.py --model_name LMM_Net --data_dir "DeepCrack" --data_name "deepcrack" --run_num 1 --epochs 100 --optimize True --validate True