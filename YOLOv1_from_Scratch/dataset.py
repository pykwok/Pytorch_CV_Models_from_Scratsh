import torch
import os
import pandas as pd
from PIL import Image

"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            csv_file,  # 训练："data/100examples.csv" # 测试："data/test.csv"
            img_dir,   # "data/images"
            label_dir, # "data/labels"
            S = 7,
            B = 2,
            C = 20,
            transform=None,):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        ''' self.annotations 内容
        img label
        000007.jpg  000007.txt
        000009.jpg  000009.txt
        '''

        # label_path 在第二列
        label_path = os.path.join(self.label_dir,
                                  self.annotations.iloc[index, 1])
        # 这张图片的GT框列表
        boxes = []
        with open(label_path) as f:
            # 每一行的内容，转换格式后，加到列表 boxes里面
            for label in f.readlines():
                # float(x) != int(float(x)) ：不相等 即 表示它原来是float

                # 1. string 转成: 原来是float的转成float（坐标x_center, y_center, w, h）、
                #                 原来是int的转成int（种类 class）
                # 2. a new line 换行符 '\n' 转成 empty space
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        # img_path 在第一列
        img_path = os.path.join(self.img_dir,
                                self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        ##################################
        # target
        # 单个cell的label格式 : [c1, c2, ..., c20, p_objct, x, y, w, h, 0, 0, 0, 0, 0]
        # 要匹配y_pred的格式，y_pred有两个anchor box ，
        # 所以 GT box label在另一个anchor的位置 都是0
        ##################################
        # （求loss用的target： loss= YOLOloss(y_pred = model(x), tartget)
        # 格式要和 y_pred 统一
        # Convert To Cells [7, 7, 30]
        label_matrix = torch.zeros((self.S,
                                    self.S,
                                    self.C + 5 * self.B))

        for box in boxes:
            # box在做完transform（和Image一起）后，
            # box的格式从tensor 转回 list
            class_label, x, y, width, height = box.tolist()
            # class的格式转成int。这个取值在{0, 1, .., 19}一共20个数
            class_label = int(class_label)

            # 看GT框属于哪个cell

            # i,j represents the cell row and cell column
            # 取整获得cell的左上角位置 (i, j)
            i, j = int(self.S * y), int(self.S * x)
            # 相对于左上角(i, j)的偏移量， ∈[0, 1]
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:

            width_pixels = (width * self.image_width)
            cell_pixels = (self.image_width)

            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels,
            simplification leads to the formulas below.
            """
            # GT框相对于这个 7*7 的特征层 缩放后的 宽、高
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object per cell !

            # cell_i,j 没有Object, 没有被taken
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                ## 标记成"taken"， 设值为1
                label_matrix[i, j, 20] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                # Set one-hot encoding for class_label
                # 独热编码
                # label : [c1, c2, ..., c20, p_objct, x, y, w, h, 0, 0, 0, 0, 0]
                # 表示种类的前20个数字“c1, .., c20”，初始值为0。
                # 设值GT class的位置的数字是1 ，其余19个class的位置是0）
                label_matrix[i, j, class_label] = 1

        return image, label_matrix