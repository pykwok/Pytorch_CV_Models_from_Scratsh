import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
load_checkpoint,
save_checkpoint,
get_loader,
check_accuracy,
save_predictions_as_imgs
)

# 超参数
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
NUM_EPOCHES = 3
NUM_WORKERS = 0
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160

PIN_MEMORY = False
LOAD_MODEL = False

TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"

def train_fn(loader,
             model,
             optimizer,
             loss_fn,
             scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        # 加一个维度
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions,
                           targets)
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqmd loop
        loop.set_postfix(loss = loss.item())



def main():
    # 定义数据增广、数据变化
    train_transform = A.Compose(
        [A.resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
         A.Rotate(limit = 35, p = 1.0),
         A.HorizontalFlip(p = 0.5),
         A.VerticalFlip(p = 0.1),
         A.Normalize(mean=[0.0, 0.0, 0.0],
                     std = [1.0, 1.0, 1.0],
                     max_pixel_value = 225.0), # 除以 225.0
         ToTensorV2(),
         ])

    val_transform = A.Compose(
        [A.resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
         A.Normalize(mean=[0.0, 0.0, 0.0],
                     std=[1.0, 1.0, 1.0],
                     max_pixel_value=225.0),
         ToTensorV2(),
         ])

    # 实例化模型，定义损失函数，优化器
    model = UNET(in_channels=3,
                 out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr = LEARNING_RATE)

    # 获取训练/验证数据集 dataloader
    train_loader, val_loader = get_loader(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pt.tar"),
                        model)

    check_accuracy(val_loader,
                   model,
                   device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    # 迭代训练
    for epoch in range(NUM_EPOCHES):
        train_fn(train_loader,
                 model,
                 optimizer,
                 loss_fn,
                 scaler)
        # 保存模型
        checkpoint = {"state_dict": model.state_dict(),
                      "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint)

        check_accuracy(val_loader,
                       model,
                       device=DEVICE)
        save_predictions_as_imgs(val_loader,
                                 model,
                                 folder="saved_images/",
                                 device=DEVICE)



if __name__ == "__main__":
    main()






















