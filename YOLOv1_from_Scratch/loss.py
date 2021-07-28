import torch
import torch.nn as nn
from utils import intersection_over_union

'''
损失函数有示性函数，分成了 I_obj 和 I_noobj：


损失函数 组成（都是 平方差Error）：
1. 平方差损失 ：（权重高） * 有物体的才会计算 x, y, w, h 的平方差损失
2. 置信度损失（在那个cell里面有box） ：
    1. 有物体的在那个cell里面 : 权重更高(5)
    2. 没有物体在那个cell里面 : 权重更低(0.5)
3. 类别损失（应该用 交叉熵 或者 negative log likelihood的，这里用的是平方差，当成回归问题）

问题：
1. [x_center, y_center, w, h]的含义是相对于左上角坐标的偏移值(YOLOv3处理方式)，还是就是anchor坐标自身？

2. 哪些是含物体的，哪些不含物体。分类标准是什么？
    - (YOLOv3处理方式)分类标准是：anchor和GT框的 IOU > IOU阈值
    
3. 示性函数怎么表示？
    1. (YOLOv3处理方式)额外加一个位置的变量p_object表示“是否含物体”，取值有三种：
        - 1  ： 正样本
        - -1 ： 负样本
        - 0  ： 要忽略的样本
        
4. 要和 预测值 一起计算损失函数的 target 怎么表示？

'''
# YoloLoss应用在 train.py
# `loss = YoloLoss(out, y)`
# 其中：`
# - out = model(x)
# - y 是 (x, y) in train_loader。y的具体内容去看 dataset.py 的 __getitem__


class YoloLoss(nn.Module):
    def __init__(self,
                 grid_cell = 7 ,  # 把图片分成  7 * 7
                 num_boxes = 2,   # 一个cell有2个bbox
                 num_class = 20): # 20个种类
        super(YoloLoss, self).__init__()
        # 论文没有求均值mean。这里遵循论文用"sum"
        self.mse = nn.MSELoss(reduction="sum")

        self.S = grid_cell
        self.B = num_boxes
        self.C = num_class

        # 对 含物体 和 不含物体的 权重
        self.lambda_coord = 5   # 含object的，权重更高
        self.lambda_noobj = 0.5 # 不含object的，权重更低

    def forward(self,
                predictions, # modell(x)的输出
                target): # dataset的__getitem__的label_matrix。前20个数是独热编码的种类
        """
        YOLOv1的输出一共有7*7 = 49个cell中，单个cell的y_pred有两个anchor，
        假设，第一个anchor可能预测的是柠檬，第二个anchor预测的是橘子。
        我们得从这两个意见不同的anchor中选出“更好的那个anchor”来代表这个cell到底预测什么。（因为一个cell只能预测一个object）
        **选择标准**： 两个anchor分别和 GT_box_label做 IOU，分数更大的那个是“更好的”，可以“负责responsible”
        """

        # YOLOv1的modell(x)的输出没有reshape，所以先reshape
        ## nn.Linear(4096, S * S * (C + B + 5))        # [7, 7, 30]
        #     [batch_size, 7 * 7 * (20 + 1 + 4 + 1 + 4)] -->
        predictions = predictions.reshape(-1,   # the number of example (batch_size)
                                          self.S, # 7
                                          self.S, # 7
                                          self.C + self.B * 5) # 20+2*5=30
        # 一共7*7=49个cell， 每个cell的信息是：
        # 1.  predictions[..., :20]   ： class probability
        # 2.1 predictions[..., 20:21] ： bbox1的置信度
        # 2.2 predictions[..., 21:25] ： bbox1的坐标（x_center_1, y_center_1, w1, h1）
        # 3.1 predictions[..., 25:26] ： bbox2的置信度
        # 3.2 predictions[..., 26:30] ： bbox2的坐标（x_center_2, y_center_2, w2, h2）

        # 一共 7*7=49个cell， 每个cell里有两个anchor，
        # 现在计算两个框的IOU，IOU值大的那个“负责” outputting that bounding box
        # predictions[..., 21:25]， 其中，`21, 22, 23, 24`是第一个边界框的四个值
        # `26, 27, 28, 29`是第二个边界框的四个值
        iou_b1 = intersection_over_union(predictions[..., 21:25],
                                         target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30],
                                         target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)],
                         dim = 0)
        # torch.max返回 最大值 和 最大值的下标arg max。
        # best_box是 单个cell 中“responsible”这个cell的预测内容的那个anchor
        iou_maxes, best_box = torch.max(ious, dim = 0)

        # 这是 identity_obj_i （取值是 0 或 1 ，取值决定于这个cell是否有object）
        # unsqueeze() 扩展维度 ：返回一个新的张量，对输入的既定位置插入维度 1
        exists_box = target[..., 20].unsqueeze(3)

        # ======================== #
        # 1 坐标损失
        # ======================== #
        # Set boxes with no object in them to 0.
        # We only take out one of the two predictions,
        # which is the one with highest Iou c
        # 如果第一个box是有物体的，则 `(1 - best_box) = 0`

        # 取出有物体的anchor的四个坐标
        box_predictions = exists_box * (
            (best_box * predictions[..., 26:30]
             + (1 - best_box) * predictions[..., 21:25])
        )

        # 取出 GT box label里的GT框坐标
        box_targets = exists_box * target[..., 21:25]

        # Take sqrt of width, height of boxes to ensure that
        # torch.sign()： 符号函数，返回一个新张量，包含输入input张量每个元素的正负
        # （大于0的元素对应1；小于0的元素对应-1；0还是0）

        # 因为可能是负数，但是开根号要取绝对值，所以后面用 sign()函数，把原本的符号弄回来
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        # 给  GT box label里的GT框 的宽和高 开方
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N, S, S, 4) --flatten--> (N*S*S, 4)
        # 计算 平方差损失
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ======================== #
        # 2.1 有物体的损失
        # ======================== #

        # respondible的那个anchor的 p_object
        pred_box = (best_box * predictions[..., 25:26]
                  + (1 - best_box) * predictions[..., 20:21])

        # (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )

        # ======================== #
        # 2.2 没有物体的损失
        # ======================== #

        # (N, S, S, 1) --> (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # ======================== #
        # 2.3 分类损失
        # ======================== #

        # # (N, S, S, 20) --> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2, ),
            torch.flatten(exists_box * target[..., :20], end_dim=-2, ),
        )


        # ======================== #
        # 3 总损失
        # ======================== #

        loss = (self.lambda_coord * box_loss
                + object_loss
                + self.lambda_noobj * no_object_loss
                + class_loss)

        return loss