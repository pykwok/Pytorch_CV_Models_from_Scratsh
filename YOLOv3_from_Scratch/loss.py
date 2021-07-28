import  torch
import torch.nn as nn
from utils import intersection_over_union


'''
在YOLOv3中，Loss分为三个部分:

1. `lbox`：[回归] -  **bounding box regression**损失。
    回归loss会乘以一个`(2 - w x h)`的比例系数，用来加大对小box的损失
   - loss_x = **self.mse_loss**(x[obj_mask], tx[obj_mask])
   - loss_y = **self.mse_loss**(y[obj_mask], ty[obj_mask])
   - loss_w = **self.mse_loss**(w[obj_mask], tw[obj_mask])
   - loss_h = **self.mse_loss**(h[obj_mask], th[obj_mask])
2. ` lobj`：**置信度**带来的误差，也就是**obj**带来的loss    
   -  前景损失：loss_conf_obj = **self.bce_loss**(pred_conf[obj_mask], tconf[obj_mask]) 
   -  背景损失：loss_conf_noobj = **self.bce_loss**(pred_conf[noobj_mask], tconf[noobj_mask])
3. `lcls`：[分类] - **类别**带来的误差 
   - loss_cls = **self.bce_loss**(pred_cls[obj_mask], tcls[obj_mask])
 
'''
# loss函数应用的地方：
# we compute the loss for a single scale.
# so, 前向传播 要调用3次这个函数

# 1、dataloader的(x, y)里的y
# y0, y1, y2 = (y[0].to(config.DEVICE),
#               y[1].to(config.DEVICE),
#               y[2].to(config.DEVICE),)
#  y0, y1, y2，分别对应于模型输出的 out[0]、out[1]、out[2]

# 2、scaled_anchors[0]、scaled_anchors[1]、scaled_anchors[2]
# 是anchor相对于 13*13、 26*26、52*52的缩放后的anchor的宽高

# 代码：
# out = model(x)
# loss_fn = YOLOv3_loss()
# loss = (loss_fn(out[0], y0, scaled_anchors[0])
#       + loss_fn(out[1], y1, scaled_anchors[1])
#       + loss_fn(out[2], y2, scaled_anchors[2])
# )

class YOLOv3_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss() # for box predictions
        self.bce = nn.BCEWithLogitsLoss()
        # for classes
        # 一个anchor只有一个class。没有multi-class。所以用交叉熵
        self.entropy = nn.CrossEntropyLoss()
        self.sigmiod = nn.Sigmoid()

        # 权重
        self.lambda_class = 1
        self.lambda_obj = 1
        self.lambda_noobj = 10
        self.lambda_box = 10


    def forward(self,
                predictions,  # 举例 out[0]
                target,       # 举例 y0
                anchors):     # 举例 scaled_anchors[0]
        # Check where obj and noobj (忽略-1的。 we ignore if target == -1)

        # in paper this is Iobj_i
        obj = target[..., 0] == 1
        # in paper this is Inoobj_i
        noobj = target[..., 0] == 0

        # ======================== #
        # 1. no object loss
        # ======================== #
        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]),
            (target[..., 0:1][noobj])
        )

        # ======================== #
        # 2.1 object loss
        # ======================== #

        # 原始anchor的shape是 3×2 ，
        # 要match the dimensions of the height and width
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        #  x, y ： b_x = sigmoid(t_x) + c_x
        # 宽, 高 ：b_w =  p_w * exp(t_w)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]),
                               self.exp(predictions[..., 3:5]) * anchors],
                              dim = -1)
        ious = intersection_over_union(box_preds[obj],
                                       target[..., 1:5][obj]).detach()

        object_loss = self.bce(self.sigmiod(predictions[..., 0:1][obj]),
                               ious * target[..., 0:1][obj]) # 乘以IOU

        # ======================== #
        # 2.2 box coordinate loss
        # ======================== #
        # x, y coordinate
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        # w, h coordinates
        target[..., 3:5] = torch.log((1e-16 + target[..., 3:5] / anchors))

        box_loss = self.mse(predictions[..., 1:5][obj],
                            target[..., 1:5][obj])

        # ================== #
        # 2.3 FOR CLASS LOSS
        # ================== #

        class_loss = self.entropy(
            (predictions[..., 5:][obj]),
            (target[..., 5][obj].long()),
        )

        # print("__________________________________")
        # print(self.lambda_box * box_loss)
        # print(self.lambda_obj * object_loss)
        # print(self.lambda_noobj * no_object_loss)
        # print(self.lambda_class * class_loss)
        # print("\n")

        return (self.lambda_box * box_loss
                + self.lambda_obj * object_loss
                + self.lambda_noobj * no_object_loss
                + self.lambda_class * class_loss
        )


