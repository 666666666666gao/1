import torch
import pickle
import os
import pickle
import numpy as np
import glob
import tifffile
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


def augmented_data_creator(image_path_list, mask_path_list, fused):
    images, masks = [], []
    for img_path, mask_path in tqdm(zip(image_path_list, mask_path_list)):
        image = np.array(Image.open(img_path)) if not fused else tifffile.imread(img_path)
        rotated_image90 = np.moveaxis(np.rot90(image, 1).copy(), [0,1,2], [2,1,0])
        rotated_image180 = np.moveaxis(np.rot90(image, 2).copy(), [0,1,2], [2,1,0])
        rotated_image270 = np.moveaxis(np.rot90(image, 3).copy(), [0,1,2], [2,1,0])

        image = np.moveaxis(image, [0,1,2], [2,1,0])

        mask = np.array(Image.open(mask_path))
        mask = torch.LongTensor(np.where(mask == True, 1, 0))
        maskss = [mask for _ in range(4)]

        images.extend([image, rotated_image90, rotated_image180, rotated_image270])
        masks.extend(maskss)

    return torch.Tensor(np.array(images)), torch.stack(masks)

if __name__ == '__main__':
    fused, img_dir, mask_dir = True, './dataset/FIND/img/fused/', './dataset/FIND/lbs/'
    image_path_list = sorted(glob.glob(os.path.join(img_dir, "*.png"))) if not fused else sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    mask_path_list = sorted(glob.glob(os.path.join(mask_dir, "*.bmp")))
    augmented_images, extended_masks = augmented_data_creator(image_path_list, mask_path_list, fused)
    pickle.dump(augmented_images, open('./dataset/FIND/preprocessed_data/augmented_images.pickle', 'wb'))
    pickle.dump(extended_masks, open('./dataset/FIND/preprocessed_data/extended_masks.pickle', 'wb'))


