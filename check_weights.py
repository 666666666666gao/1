import torch

# 加载权重文件
state = torch.load('saved_models/LMM_Net/best_model_num_1.pt', map_location='cpu')

# 打印权重字典的键
print("Model state dict keys:")
print(state['model_state_dict'].keys())

# 打印损失值
if 'val_loss' in state:
    print("\nValidation loss:", state['val_loss'])
elif 'train_loss' in state:
    print("\nTraining loss:", state['train_loss']) 