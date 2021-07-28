import torch
import torch.nn as nn

import numpy as np

# 由论文的网络结构图，写出来的网络配置（只含Darknet19的卷积部分，没有后面的FC层）
# Tuple: (kernel_size, filter_of_output, stride, padding)
# Str  : "M", Max pooling
# List :  Tuples 和 最后一个字（它表示"重复多少次"）

architecture_config = [
    # Tuple: (kernel_size, filter_of_output, stride, padding)
    (7, 64, 2, 3),
    # Str : Max pooling
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    # List :
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],  # 最后一个字表示 重复四次
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],  # 最后一个字表示 重复二次
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 **kwargs): # keyword arguments
        super(CNNBlock, self).__init__() # initialize the parent class
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              bias = False, # 要用BN层，所以设置为False
                              **kwargs)
        # YOLOv1发表的时候还没有BN层，给它加上去
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class YOLOv1(nn.Module):
    def __init__(self, in_channels = 3, **kwargs):
        super(YOLOv1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layer(self.architecture)
        self.fcs = self._creare_fcs(**kwargs) 

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim = 1))

    def _create_conv_layer(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(in_channels,
                             out_channels = x[1],
                             kernel_size = x[0],
                             stride = x[2],
                             padding = x[3])
                ]

                in_channels = x[1]

            elif type(x) == str:
                 layers += [
                     nn.MaxPool2d(kernel_size=(2, 2),
                                  stride=(2, 2))
                 ]
            elif type(x) == list:
                # Tuple
                conv1 = x[0]
                # Tuple
                conv2 = x[1]
                # Integer
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(in_channels,
                                 out_channels = conv1[1],
                                 kernel_size = conv1[0],
                                 stride = conv1[2],
                                 padding = conv1[3],
                                 )
                    ]
                    layers += [
                        CNNBlock(in_channels = conv1[1],
                                 out_channels= conv2[1],
                                 kernel_size= conv2[0],
                                 stride = conv2[2],
                                 padding = conv2[3],
                                 )
                    ]

                    in_channels = conv2[1]
        # "nn.Sequential(*layers)" 是 unpack this list
        # and convert it to an nn.Sequential
        return nn.Sequential(*layers)

    def _creare_fcs(self,
                    split_size,   # YOLO划分网格的大小， 举例，7 * 7
                    num_boxes,    # 一个cell输出两个bboxes
                    num_classes): # 种类
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(nn.Flatten(),
                             nn.Linear(1024 * S * S, 4096), # 原始论文是4096，省时间写成496
                             nn.Dropout(0.0),
                             nn.LeakyReLU(0.1),
                             nn.Linear(4096, S * S * (C + B + 5)) # [7, 7, 30]
                             )


# 测试一下YOLO结构
def test(split_size = 7,
         num_boxes = 2,
         num_classes = 20
         ):
    model = YOLOv1(split_size = split_size,
                   num_boxes = num_boxes,
                   num_classes = num_classes)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)


# test()
# 输出 torch.Size([2, 1323])
























