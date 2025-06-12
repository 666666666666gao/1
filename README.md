# Crack Segmentation

基于深度学习的裂缝分割项目，支持多种模型架构和训练策略。

## 支持的模型

- UNet
- LMM_Net
- EfficientCrackNet

## 环境要求

- Python 3.6+
- PyTorch 1.7+
- CUDA (推荐用于GPU加速)
- 其他依赖包：
  - torchvision
  - numpy
  - opencv-python
  - scikit-learn
  - optuna (用于超参数优化)
  - matplotlib
  - tqdm

## 项目结构

```
Crack-Segmentation/
├── main_dev.py          # 主训练脚本
├── predict.py           # 预测脚本
├── model.py            # 模型定义
├── efficientnet.py     # EfficientCrackNet模型
├── utils.py            # 工具函数
├── loss_functions.py   # 损失函数
├── eval_metrics.py     # 评估指标
├── data_preprocessing.py # 数据预处理
├── saved_models/       # 保存的模型权重
├── predictions/        # 预测结果
├── plots/             # 训练曲线图
└── DeepCrack/         # 数据集目录
    ├── train_img/     # 训练图像
    ├── train_lab/     # 训练标签
    ├── valid_img/     # 验证图像
    ├── valid_lab/     # 验证标签
    ├── test_img/      # 测试图像
    └── test_lab/      # 测试标签
```

## 功能特点

1. **多种模型支持**
   - UNet: 经典的分割网络
   - LMM_Net: 轻量级分割网络
   - EfficientCrackNet: 基于EfficientNet的改进网络

2. **训练功能**
   - 支持验证集训练
   - 早停机制
   - 学习率自动调整
   - 损失函数组合（BCE + Dice + IoU）
   - 训练过程可视化
   - 模型权重自动保存

3. **超参数优化**
   - 使用Optuna进行自动超参数搜索
   - 支持优化学习率、批量大小、损失权重等
   - 支持多种优化器和学习率调度器

4. **预测功能**
   - 支持批量预测
   - 自动参数搜索
   - 参数保存和加载
   - 后处理优化

## 使用方法

### 1. 训练模型

基本训练命令：
```bash
python main_dev.py --model_name UNet --data_dir "DeepCrack" --data_name "deepcrack" --run_num 1 --epochs 100 --validate True
```

带超参数优化的训练：
```bash
python main_dev.py --model_name UNet --data_dir "DeepCrack" --data_name "deepcrack" --run_num 1 --epochs 100 --optimize True --validate True
```

参数说明：
- `--model_name`: 选择模型 (UNet/LMM_Net/EfficientCrackNet)
- `--data_dir`: 数据集目录
- `--data_name`: 数据集名称
- `--run_num`: 运行编号
- `--epochs`: 训练轮数
- `--validate`: 是否使用验证集
- `--optimize`: 是否进行超参数优化

### 2. 预测

基本预测命令：
```bash
python predict.py --model_name UNet --model_path "saved_models/UNet/best_model_num_1.pt" --input_dir "predict" --output_dir "predictions/UNet"
```

自动搜索最佳参数：
```bash
python predict.py --model_name UNet --model_path "saved_models/UNet/best_model_num_1.pt" --input_dir "DeepCrack/test_img" --mask_dir "DeepCrack/test_lab" --auto_search --output_dir predictions/UNet
```

强制重新搜索参数：
```bash
python predict.py --model_name UNet --model_path "saved_models/UNet/best_model_num_1.pt" --input_dir "DeepCrack/test_img" --mask_dir "DeepCrack/test_lab" --auto_search --force_search --output_dir predictions/UNet
```

### 3. 最佳参数参考

EfficientCrackNet:
- threshold: 0.5
- min_area: 50
- morph_kernel: 1

UNet:
- threshold: 0.4
- min_area: 120
- morph_kernel: 5

## 注意事项

1. 确保数据集目录结构正确
2. 训练时建议使用GPU加速
3. 超参数优化可能需要较长时间
4. 预测时会自动保存最佳参数到 `best_params` 目录

## 更新日志

### 最新更新
- 添加了参数自动搜索和保存功能
- 优化了训练过程中的内存使用
- 改进了早停机制
- 添加了更多的数据增强方法

## 许可证

MIT License
