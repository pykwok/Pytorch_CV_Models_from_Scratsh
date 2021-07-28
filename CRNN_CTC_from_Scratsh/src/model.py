import torch
from torch import nn
from torch.nn import functional as F


class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaModel, self).__init__()
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        # 有 75 time step and for each time step, you have 1152 values
        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2))
        # 有 75 time step and for each time step, you have 64 values
        # self.linear_1 = nn.Linear(1152, 64) # 输入为torch.rand((2, 3, 70, 300))
        self.linear_1 = nn.Linear(1600, 64) # 输入为torch.rand((2, 3, 100, 300))

        self.drop_1 = nn.Dropout(0.2)

        self.lstm = nn.GRU(64,
                           32,
                           bidirectional = True,
                           num_layers = 2,
                           dropout = 0.25,
                           batch_first = True)
        self.output = nn.Linear(64, num_chars + 1)

    def forward(self,
                images,
                targets=None): # inference的时候没有target
        # bs, c, h, w = images.size()
        bs, _, _, _ = images.size()

        x = F.relu(self.conv_1(images))
        print(x.size()) # torch.Size([2, 128, 100, 300])
        x = self.pool_1(x)
        print(x.size()) # torch.Size([2, 128, 50, 150])

        x = F.relu(self.conv_2(x))
        print(x.size()) # torch.Size([2, 64, 50, 150])
        x = self.pool_2(x)
        # torch.Size([2, 64, 25, 75])
        print(x.size())  # torch.Size([2, 64, 25, 75])

        # 因为用RNN的时候要用width
        #  [2, 64, 25, 75] --> [2, 75, 64, 25]
        x = x.permute(0, 3, 1, 2)
        print(x.size())
        # x.size(1)是 width 。例子中是 75
        x = x.view(bs, x.size(1), -1)
        # 拉平后，[2, 75, 1600] 其中， 1600 = 64 * 25
        print(x.size()) # [2, 75, 1600]

        x = F.relu(self.linear_1(x))
        x = self.drop_1(x)
        print(x.size()) # [2, 75, 64]

        x, _ = self.lstm(x)
        print("x.size()", x.size()) # [2, 75, 64]
        print("_.size()", _.size()) # [4, 2, 32]

        # vector_size =  20
        x = self.output(x)
        print(x.size()) # [2, 75, 20]

        # time step放在第一，batch_size放在第二
        x = x.permute(1, 0, 2)
        print(x.size()) # [75, 2, 20]

        # 训练模式，target不为None
        if targets is not None:
            log_probs = F.log_softmax(x, 2)
            print("log_probs.shape", log_probs.shape) # torch.Size([75, 2, 20])

            input_lengths = torch.full(
                size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32
            )
            print("input_lengths", input_lengths) # tensor([75, 75], dtype=torch.int32)
            print("input_lengths.shape", input_lengths.shape) # torch.Size([2])

            target_lengths = torch.full(
                size=(bs,), fill_value=targets.size(1), dtype=torch.int32
            )
            print("target_lengths", target_lengths) # tensor([5, 5], dtype=torch.int32)
            print("target_lengths.shape", target_lengths.shape) #  torch.Size([2])

            loss = nn.CTCLoss(blank=0)(
                log_probs,
                targets,
                input_lengths,
                target_lengths
            )
            print("loss", loss) # tensor(39.7398, grad_fn=<MeanBackward0>)
            print("loss.shape", loss.shape) # torch.Size([])

            # 训练的时候，target != None ，会返回loss。下面的推理的时候，不返回loss
            return x, loss

        # 推理不返回loss
        # predictions 和 Loss （其中x是predictions)
        return x, None


if __name__ == "__main__":
    # print(lbl_enc.classes_) # ['2' '3' '4' '5' '6' '7' '8' 'b' 'c' 'd' 'e' 'f' 'g' 'm' 'n' 'p' 'w' 'x' 'y']
    # print(len(lbl_enc.classes_)) # 19
    cm = CaptchaModel(19)
    # img = torch.rand((1, 3, 75, 300))
    img = torch.rand((2, 3, 100, 300))
    # target = torch.randint(1, 20, (1, 5))
    # x, loss = cm(img, target)

    # x, _ = cm(img, torch.randint(1, 20, (1, 5)))
    # x, _ = cm(img, torch.rand((1, 5)))
    x, _ = cm(img, torch.rand((2, 5)))

