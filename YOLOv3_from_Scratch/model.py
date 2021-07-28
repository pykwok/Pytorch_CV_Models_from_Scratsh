import torch
import torch.nn as nn

# 1. Tuplel: (out_channels, kernel_size, stride)
# 2. List  : -- "B" : 论文上的作者圈起来的“残差卷积块” "B" indicating a residual block
#            -- "1" : 重复的次数
# 3. Str   : -- "S" is for “scale prediction” block and computing the yolo loss
#            -- "U" is for upsampling the feature map and concatenating with a previous layer

# 53层Darknet + 53层其它的 = 106层
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # 上面的截止到这里是Darknet53的内容。 To this point is Darknet-53
    (512, 1, 1),   # 1×1卷积
    (1024, 3, 1),
    "S", # 输出层一（在第82层）。下采样倍率 / network stride：32
    (256, 1, 1),   # 1×1卷积
    "U",
    (256, 1, 1),   # 1×1卷积
    (512, 3, 1),
    "S", # 输出层二（在第94层）。下采样倍率 / network stride：16
    (128, 1, 1),   # 1×1卷积
    "U",
    (128, 1, 1),   # 1×1卷积
    (256, 3, 1),
    "S", # 输出层三（在第106层）。下采样倍率 / network stride：8
]


class CNNBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_act=True,  # 因为输出层scale layer不要用batchnorm
                 **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              bias=not bn_act,  # 如果用了BN，那bias是一个unnecessary parameter
                              **kwargs) # kernel_size、stride、padding等
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)

# 残差块
# ["B", 1],..., ["B", 2],...,["B", 8],...,["B", 8],...,["B", 4],
class ResidualBlock(nn.Module):
    def __init__(self,
                 channels,
                 use_residual=True,
                 num_repeats = 1):
        super().__init__()
        self.layers = nn.ModuleList()

        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels,      # 论文的网络结构 "TyPE类型: convolution卷积、Filters个数: 32、卷积核大小 Size:1×1"
                             channels // 2, # 论文的网络结构 "TyPE类型: convolution卷积、Filters个数: 64、卷积核大小 Size:3×3"
                             kernel_size=1), # padding的默认值是 0
                    CNNBlock(channels // 2,
                             channels,
                             kernel_size=3,
                             padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            # 版本一：
            if self.use_residual: # 如果有残差链接
                x = x + layer(x)
            else:
                x = layer(x)
            # 版本二：
            # x = x + layer(x) if self.use_residual else layer(x)

        return x

# 对于Batch_size张里的其中一张图片来说，经过网络后有三个输出层。
# 每层的输出是由 1×1的卷积得到的。卷积个数为： (4 + 1 + num_class) *  num_anchor_box。
# 举例COCO数据集，(4+1+80)*3 = 255。
# 即，COCO数据集的 输出层，是由 255个1×1的卷积 得到的。
#     COCO数据集的三个输出层的输出shape分别为： [Batch_size, 13, 13, 255]、[Batch_size, 26, 26, 255] 和 [Batch_size, 52, 52, 255]
# ----------------------------------------------
# |  anchor_box1 |  anchor_box2 |  anchor_box1 |
# ----------------------------------------------
# 内容如下：
# [tx_anchor1, ty_anchor1, th_anchor1, tw_anchor1, Prob_object_anchor1, P_class1_anchor1, ..., P_NumOfClass_anchor1,
#  tx_anchor2, ty_anchor2, th_anchor2, tw_anchor2, Prob_object_anchor2, P_class1_anchor2, ..., P_NumOfClass_anchor2,
#  tx_anchor3, ty_anchor3, th_anchor3, tw_anchor3, Prob_object_anchor3, P_class1_anchor3, ..., P_NumOfClass_anchor3,]

class ScalePrediction(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes):
        super().__init__()
        # 3个anchor box, 每个的内容:[tx, ty, th, tw, Prob_object_anchor1, P_class1, ..., P_NumOfClass]
        self.pred = nn.Sequential(
            CNNBlock(in_channels,
                     2 * in_channels,
                     kernel_size = 3,
                     padding = 1),
            CNNBlock(in_channels * 2,
                     3 * (4 + 1 + num_classes), # 3 * (4 + 1 + 80) = 255
                     bn_act = False,
                     kernel_size = 1)   # 1×1卷积的用途
        )

        self.num_classes = num_classes # COCO数据集的是 80

    def forward(self, x):
        # scale_cell_number有3种： 13 、26、 52
        # 下面举例 scale_cell_number == 13
        # reshape()后的shape是 ：[batch_size, 3, 85, 13, 13]
        # permute()后的shape是 ：[batch_size, 3, 13, 13, 85]
        return (self.pred(x)
                .reshape(x.shape[0], 3, 5 + self.num_classes, x.shape[2], x.shape[3])
                .permute(0, 1, 3, 4, 2))

class YOLOv3(nn.Module):
    def __init__(self,
                 in_channels = 3,
                 num_classes = 80):
        super().__init__()
        self.num_classes = num_classes # 80
        self.in_channels = in_channels # 3
        self.layers = self._create_conv_layers()

    def forward(self, x):
        # for三种scale。
        outputs = []
        # 路由层 (skip connection里要和uosampling结果做concat的对象)
        # 它是 ScalePrediction的前一层acnnblock的输出
        route_connections = []

        for layer in self.layers:
            # isinstance == 输出层
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                # ScalePrediction是一个分支，我们希望continue主线下去，所以要come back
                # x走两层分支：
                # 1. 输出层分支
                # 2. 继续走主线（上采样，前一层的同尺寸的特征层concate后，再卷积。再走两个分支）
                # 所以要用 "continue"
                continue

            x = layer(x)

            # isinstance == 残差块
            # layer.num_repeats == 8 的残差块输出 是路由层
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            # isinstance == 上采样 （一共只有两个上采样）
            # Upsample后要做Skip connection
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]],
                              dim=1) # concat along dim=1 for the channels
                # pop掉
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            # isinstance == 卷积
            if isinstance(module, tuple):
                # 拆来元组的三个变量
                out_channels, kernel_size, stride = module

                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding = 1 if kernel_size == 3 else 0, # kernel_size == 1的时候，padding = 0
                    )
                )

                in_channels = out_channels

            # isinstance == 残差卷积块
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels,
                                            num_repeats = num_repeats,))

            # isinstance == 输出层"S" 或者 上采样"U"
            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels,
                                      use_residual=False,
                                      num_repeats = 1),
                        CNNBlock(in_channels,
                                 in_channels // 2,
                                 kernel_size=1),
                        ScalePrediction(in_channels // 2,
                                        num_classes = self.num_classes),
                    ]

                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),) # 上采样倍率为 2
                    # 上采样后要和之前的路由层concat， 所有要改变channel个数
                    # 一共两次上采样：
                    #第一次的channel：256。它和第一个ScalePrediction之前的那个CNNBlock的输出(channel值512)做connect。256 + 512 = 768 = 256 * 3
                    #第二次的channel：128。它和第二个ScalePrediction之前的那个CNNBlock的输出(channel值512)做connect。128 + 256 = 384 = 128 * 3
                    in_channels = in_channels * 3

        return layers

if __name__ == "__main__":
    num_classes = 80
    IMEGE_SIZE = 416

    model = YOLOv3(num_classes = num_classes)
    x = torch.randn((2, 3, IMEGE_SIZE, IMEGE_SIZE))
    out = model(x)
    print(model)

    assert model(x)[0].shape == (2, 3, IMEGE_SIZE//32, IMEGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMEGE_SIZE//16, IMEGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMEGE_SIZE//8, IMEGE_SIZE//8, num_classes + 5)

    print("success!")










