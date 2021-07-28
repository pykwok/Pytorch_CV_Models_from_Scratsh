import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self,
                 in_channels = 3,
                 out_channels = 1,
                 features = [64, 128, 256, 512]):
        super(UNET, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #---------------------------
        # 一、 contracting path 收缩分支
        # --------------------------
        self.downs = nn.ModuleList()

        for feature in features:
            self.downs.append(DoubleConv(in_channels=in_channels,
                                         out_channels=feature))
            in_channels = feature

        # --------------------------
        # 二、 底部连接
        # --------------------------
        self.bottleneck = DoubleConv(in_channels = features[-1], # 512 --> 1024
                                     out_channels = features[-1] * 2)

        # --------------------------
        # 三、 expansive path 扩张分支
        # --------------------------
        self.ups = nn.ModuleList()

        # 上采样、concat、2个卷积 --> 上采样、concat、2个卷积 --> ...
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, # 第一次是 28 --> 54
                                               feature,
                                               kernel_size=2,
                                               stride=2
                                               )
                            )
            self.ups.append(DoubleConv(in_channels = feature * 2, # 因为有Skip gram
                                        out_channels = feature)
                            )
        # --------------------------
        # 四、 最后输出层
        # --------------------------
        self.final_conv = nn.Conv2d(in_channels=features[0],
                                    out_channels = out_channels,
                                    kernel_size=1)

    def forward(self, x):
        # 存放skip的内容
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            # 下采样
            x = self.pool(x)
        # [3, 512, 35, 35] --> [3, 1024, 35, 35]
        x = self.bottleneck(x)
        # 颠倒一下顺序
        skip_connections = skip_connections[::-1]

        # print(len(self.ups)) # 8
        # idx的取值：0、2、4、6
        for idx in range(0, len(self.ups), 2):
            # print(idx)
            # idx=0, [3, 1024, 35, 35] --> [3, 512, 70, 70]
            x = self.ups[idx](x)
            # skip_connections里面有4项内容，shape分别为 ：
            # [3, 512, 71, 71]、[3, 256, 142, 142]、[3, 128, 284, 284]、[3, 64, 568, 568]
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x,
                              size=skip_connection.shape[2:])
            # idx=0, shape=[3, 1024, 71, 71]
            # idx=2, shape=[3, 512, 142, 142]
            concat_skip = torch.cat((skip_connection,
                                     x),
                                    dim = 1
                                    )
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 568, 568))
    model = UNET(in_channels=1,
                 out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()