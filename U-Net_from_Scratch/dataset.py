import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CarvanaDataset(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 transform = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir + self.images[index])
        mask_path = os.path.join(self.mask_dir + self.images[index])

        image = np.array(Image.open(image_path).convert("RGB"))
        # “L" ：因为 mask是greyscale
        mask = np.array(Image.open(mask_path).convert("L"),
                        dtype = np.float32)
        # 因为要用sigmiod来计算损失值。
        #  mask转为二值。取值只有：0 或 1
        mask[mask == 225.0] = 1.0

        if self.transform is not None:
            augmentation = self.transform(image = image,
                                          mask = mask)
            image = augmentation["image"]
            mask = augmentation["mask"]

        return image, mask