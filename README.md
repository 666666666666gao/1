# Crack Segmentation 项目说明

本项目基于深度学习实现混凝土裂缝分割，支持多种主流模型，具备自动化训练、超参数优化、批量预测、自动评估、丰富可视化和详细结果分析，适合科研、工程、课程设计等多场景。

---

## 📁 目录结构与功能说明

| 文件/文件夹         | 主要功能与细节 |
|---------------------|-------------------------------------------------------------------|
| main_dev.py         | 训练主入口，支持超参数优化、早停、断点续训、自动保存最优模型、训练曲线可视化 |
| predict.py          | 批量预测与自动参数搜索，支持最佳参数保存/加载、后处理、结果可视化           |
| test.py             | 多模型对比测试，随机抽样、评估指标展示、掩码可视化                        |
| eval.py             | 批量评估，输出各类分割指标（F1、IoU等）                                   |
| model.py            | UNet、LMM_Net等模型结构定义                                               |
| efficientnet.py     | EfficientCrackNet模型结构实现                                            |
| mobileVitblock.py   | MobileViT相关模块实现                                                    |
| utils.py            | 数据集加载、数据增强、工具函数、权重初始化等                              |
| loss_functions.py   | BCE、Dice、IoU等损失函数实现                                             |
| eval_metrics.py     | 评估指标函数（F1、IoU等）                                                |
| data_preprocessing.py | 数据预处理脚本（如格式转换、切片等）                                   |
| sample_img_extraction.py | 样本图片提取脚本                                                    |
| check_weights.py    | 权重文件检查工具                                                        |
| DeepCrack/          | 数据集目录，含训练/验证/测试图片及掩码                                   |
| saved_models/       | 训练过程中自动保存的模型权重                                             |
| predictions/        | 预测结果输出目录                                                        |
| plots/              | 训练/验证曲线、流程图、结果可视化                                       |
| best_params/        | 自动参数搜索保存的最佳参数json                                           |
| .gitignore          | Git忽略配置，防止大文件/中间结果上传                                     |

---

## 🌟 EfficientCrackNet 模型架构详解

EfficientCrackNet 是本项目的高效裂缝分割主力模型，融合了 EfficientNet 的高效特征提取能力与轻量化解码结构，适合边缘设备和大规模工程应用。

### EfficientCrackNet 结构流程图
```mermaid
graph TD
    A[输入图像] --> B[EfficientNet主干特征提取]
    B --> C[多尺度特征融合]
    C --> D[轻量解码器（上采样+卷积）]
    D --> E[分割头（1x1卷积+Sigmoid）]
    E --> F[输出掩码]
```
- **主干网络**：采用 EfficientNet（如B0/B1）提取多层特征，兼顾速度与精度。
- **多尺度融合**：融合不同层次的特征，提升对细小裂缝的感知。
- **轻量解码器**：采用上采样+卷积结构，逐步恢复空间分辨率。
- **分割头**：1x1卷积输出单通道概率图，Sigmoid激活得到像素级分割。

### EfficientCrackNet 主要优点
- 计算量小，推理速度快，适合大批量或边缘部署。
- 多尺度特征融合，提升对细微裂缝的检测能力。
- 结构简洁，易于扩展和迁移。

---

## 🚦 训练、预测、评估流程图与简介

### 1. 训练流程
```mermaid
flowchart TD
    A[加载数据集] --> B[数据增强/预处理]
    B --> C[模型初始化]
    C --> D{是否加载预训练权重}
    D --是--> E[加载权重]
    D --否--> F[随机初始化]
    E & F --> G[训练循环]
    G --> H[验证集评估]
    H --> I{早停/保存最优模型}
    I --满足条件--> J[保存权重/曲线]
    I --未满足--> G
    J --> K[训练结束]
```
- 支持超参数优化、早停、断点续训、自动保存最优模型。
- 训练过程自动保存最优模型和训练曲线。

### 2. 预测流程
```mermaid
flowchart TD
    A[加载模型与权重] --> B[加载图片]
    B --> C{是否自动参数搜索}
    C --是--> D[遍历参数组合]
    D --> E[推理+评估]
    E --> F[保存最佳参数]
    C --否--> G[直接推理]
    F & G --> H[后处理+保存掩码]
    H --> I[输出可视化结果]
```
- 自动搜索最佳后处理参数，支持批量预测和可视化。

### 3. 评估流程
```mermaid
flowchart TD
    A[加载预测掩码与GT] --> B[批量评估]
    B --> C[输出F1/IoU等指标]
    C --> D[生成评估报告]
```
- 支持批量评估，输出分割指标和可视化报告。

---

## 🧮 损失函数与评估指标详解

### 1. 二值交叉熵（BCE）
用于像素级二分类，衡量预测概率与真实标签的差异。

公式：
$$
\text{BCE} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right]
$$
其中 $y_i$ 为真实标签，$p_i$ 为预测概率。

### 2. Dice Loss（Dice系数损失）
衡量预测掩码与真实掩码的重叠程度，适合处理类别极不平衡的分割任务。

公式：
$$
\text{Dice} = \frac{2 \sum_i p_i y_i}{\sum_i p_i + \sum_i y_i}
$$
Dice Loss $= 1 - \text{Dice}$

### 3. IoU Loss（交并比损失）
衡量预测与真实掩码的交集与并集比例。

公式：
$$
\text{IoU} = \frac{\sum_i p_i y_i}{\sum_i p_i + \sum_i y_i - \sum_i p_i y_i}
$$
IoU Loss $= 1 - \text{IoU}$

### 4. F1 Score（F1分数）
综合考虑精确率（Precision）和召回率（Recall），是分割常用的综合指标。

公式：
$$
\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$
其中：

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

### 5. 评估流程图
```mermaid
graph TD
    A[预测掩码] --> B[与GT对齐]
    B --> C[计算TP/FP/FN]
    C --> D[计算F1、IoU、Precision、Recall]
    D --> E[输出评估报告/表格]
```

---

## 🚀 主要功能流程

### 1. 总体流程概览
```mermaid
flowchart TD
    数据准备 -->|数据增强| 训练
    训练 -->|保存最优模型| 预测
    预测 -->|自动参数搜索| 评估
    评估 -->|生成报告| 可视化
    可视化 -->|对比分析| 结果总结
```

### 2. 训练、预测、评估等详细流程见上文

---

## 📊 结果与评估

- **训练过程**：每轮训练/验证损失、F1、IoU等指标自动保存并绘图（见 plots/）。
- **预测结果**：每张图片的分割掩码自动保存（见 predictions/），支持原图与掩码对比可视化。
- **多模型对比**：test.py 支持同一图片多模型分割效果对比，指标一目了然。
- **评估报告**：eval.py 可批量输出所有图片的分割指标（F1、IoU、Precision、Recall），可导出为表格或可视化图。
- **参数最优性**：predict.py 自动搜索并保存最佳后处理参数，保证分割精度最大化。

---

## 🛠️ 依赖环境
- Python 3.6+
- PyTorch 1.7+
- torchvision, numpy, opencv-python, scikit-learn, optuna, matplotlib, tqdm
- 推荐用 requirements.txt 管理依赖

---

## 📖 快速上手

### 训练
```bash
python main_dev.py --model_name UNet --data_dir "DeepCrack" --data_name "deepcrack" --run_num 1 --epochs 100 --validate True
```

### 超参数优化训练
```bash
python main_dev.py --model_name UNet --data_dir "DeepCrack" --data_name "deepcrack" --run_num 1 --epochs 100 --optimize True --validate True
```

### 预测（自动参数搜索）
```bash
python predict.py --model_name UNet --model_path "saved_models/UNet/best_model_num_1.pt" --input_dir "DeepCrack/test_img" --mask_dir "DeepCrack/test_lab" --auto_search --output_dir predictions/UNet
```

### 评估
```bash
python eval.py --model_name UNet --pred_dir predictions/UNet --gt_dir DeepCrack/test_lab
```

### 多模型对比
```bash
python test.py
```

---

## 📝 许可证
MIT License

---

> 本项目适合科研、工程、课程设计等多场景，欢迎二次开发与贡献！

---

如需更详细的 API 文档、参数说明或自定义流程图，欢迎在 Issues 区留言或提交 PR！
