import config
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageFile
from utils import (
    iou_width_height as iou,
    non_max_suppression as nms,
)

# https://sannaperzon.medium.com/yolov3-implementation-with-training-setup-from-scratch-30ecb9751cb0

'''
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  

ANCHORS.shape =  [3, 3, 2]

# Note these have been rescaled to be between [0, 1]
'''


ImageFile.LOAD_TRUNCATED_IMAGES = True

# For each bounding box
# we will then assign it to the "grid cell" which contains its midpoint
#          and decide "which anchor is responsible for it"
#               by determining which anchor the bounding box has highest intersection over union with.
class YOLODataset(Dataset):
    def __init__(self,
                 csv_file,
                 img_dir, label_dir,
                 anchors,
                 image_size = 416,
                 S = [13, 26, 52],
                 num_class = 20,
                 transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.C = num_class
        self.img_dir = img_dir
        self.label_dir = label_dir

        # for all 3 scales
        #  combine the list above to a tensor of shape (9,2) corresponding to each anchor box on all scales
        # shape : [3, 3, 2] --> [9, 2]
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape(0) # 9
        # 每层scale层都有 3种anchor
        self.num_anchors_per_scale = self.num_anchors // 3  # 9//3 = 3

        self.S = S
        self.img_size = image_size
        self.transform = transform

        # "set one anchor responsible for each cell in all of different scales"
        # responsible的那个box，是和 GT框的 IOU 最大的那个
        self.ignore_iou_thresh = 0.5

    def __getitem__(self, index):
        # self.annotations内容。第一列：img_path、第二列：label_path

        # 标签
        # self.annotations.iloc[index, 1] 里面的数字"1"是因为在第二列
        label_path = os.path.jion(self.label_dir,
                                  self.annotations.iloc[index, 1])
        # GT框
        # yolov3的数据的格式是： [class , x, y, w, h].
        # 但是做数据增广，标签和图片一起变化，那个包要求的格式是 [x, y, w, h, class]
        # 函数 numpy的tolist() ：矩阵 ——> 列表
        bboxes = np.roll(np.loadtxt(fname = label_path,
                                    delimiter = " ",
                                    ndim = 2),
                         4,
                         axis = 1).tolist()
        # 图像
        # self.annotations.iloc[index, 1] 里面的数字"0"是因为在第一列
        img_path = os.path.join(self.img_dir,
                                self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            # 图片 和 标签 一起放进去，做旋转、翻转等处理
            augmentation = self.transform(image = image,
                                          bboxes = bboxes)
            image = augmentation["image"]
            bboxes = augmentation["bboxes"]

        #####################
        # Building targets
        #####################
        # When we load the labels for a specific image it will only be an array with all the bounding boxes and to be able to calculate the loss
        # we want to format the targets similarily to the model output.
        # The model will output predictions on three different scales so we will also build three different targets.

        # [p_obj, x, y, w, h, num_class]

        # [3, 13, 13, 6] 其中，6 : [p_obj, x, y, w, h, num_class]
        # targets先全部置为0。后面要修改
        # 1. 正样本 (有object的):
        # targets[scale_idx][anchor_on_scale, i, j, 0] = 1
        # targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
        # targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
        # has_anchor[scale_idx] = True
        # 2. 舍弃的样本（没有object，且IOU又很高的 ，属于难区分的）:
        # targets[scale_idx][anchor_on_scale, i, j, 0] = -1
        # 3. 负样本（没有object，而且是背景那一类的，容易区分的负样本）
        # 默认值0 ：targets[scale_idx][anchor_on_scale, i, j, 0] = 0

        targets = [torch.zeros((self.num_anchors // 3, # 9//3 = 3 # 每层scale层都有3种anchor
                              S, # 13、26 或者 52
                              S,
                              6)) for S in self.S]

        # GT框在三个scale层都可以计算缩放比例得到一个框。

        # 对于三个scale层来说，
        # 我们要指定which anchor应该responsible 和 which cell应该responsible
        # 通过“which anchor有最高IOU”来指定
        for box in bboxes:
            # 先计算IOU值，根据IOU值来判断哪个 responsible
            # GT框循环里，每个GT框都 和 9个anchor 计算IOU
            iou_anchors = iou(torch.tensor(box[2:4]), # GT框的宽高
                              self.anchors) # anchor的宽高
            # 第一个 是 best anchor
            anchor_indices = iou_anchors.argsort(descending = True,
                                                 dim = 0)
            # GT框数据
            x, y, width, height, class_label = box
            has_anchor = [False, False, False]

            #  然后我们将遍历九个索引以将目标分配给最佳锚点
            #  loop through the nine indices to assign the target to the best anchors

            # 我们的目标是将每个目标边界框分配给每个尺度上的一个锚点，即总而言之，将每个目标分配给我们上面初始化的每个目标矩阵中的一个锚点。
            # 此外，我们还将检查一个锚点是否不是最适合边界框的，但它仍然具有高于 0.5 的iou，然后我们将标记此目标，以便不会产生任何损失 这个锚框的预测。
            #  Our goal is to assign "each target bounding box" to "an anchor" on each scale i.e.
            #  in total assign each target to one anchor in each of the target matrices we intialized above.
            #  In addition we will also check if an anchor is not the most suitable for the bounding box but it still has an intersection over union higher than 0.5
            #  and then we will mark this target such that no loss is incurred for the prediction of this anchor box.

            # 一共有3个输出层，每层对应三种anchor。
            # 一共九种anchor
            for anchor_idx in anchor_indices:
                # scale_idx ∈ {0, 1, 2} 这个anchor属于第几个scale
                scale_idx = anchor_idx // self.num_anchors_per_scale
                # anchor_on_scale ∈ {0, 1, 2}。在第scale_idx个scale层上的第几个anchor
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                # 看第scale_idx个输出层有多少个cell（13、 26、 52）
                S = self.S[scale_idx]
                # x, y ∈ [0,1]区间的数。是"相对于原图尺寸的"
                # i, j 是计算这个cell在网格的位置。（含物体的那个cell的左上角坐标）
                i, j = int(S * y), int(S * x)  # which cell

                # 防止两个object是同一个bounding box。
                # 所以先取出来，然后看之前 有没有被taken过
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    # 记为 1
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1

                    # 相对于cell左上角的 delta_x 和 delta_y
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]

                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell

                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )

                    # 修改这个框在target的内容
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    #  标记一下，纪录taken状态
                    has_anchor[scale_idx] = True

                # 要被ignored的记为 -1
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    # 置为 -1
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

            return image, tuple(targets)

    def __len__(self):
        return len(self.annotations)



'''
def test():
    anchors = config.ANCHORS

    transform = config.test_transforms

    dataset = YOLODataset(
        "COCO/train.csv",
        "COCO/images/images/",
        "COCO/labels/labels_new/",
        S=[13, 26, 52],
        anchors=anchors,
        transform=transform,
    )
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            print(anchor.shape)
            print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        print(boxes)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)


if __name__ == "__main__":
    test()
'''