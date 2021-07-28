import torch
import torchvision
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#启动tensorboard
#cd/d F:\Deep_Learning_model_from_Scratch\DCGAN_from_Scratsh
#conda activate unet38
#tensorboard --logdir=logs


# 100 -> 4*4*1024 -> 8*8*512 -> 16*16*256 -> 32*32*128 -> 64*64*3

# 超参数
NOISE_DIM = 256 # 论文是100
IMAGE_SIZE = 64 # MINIST的数据是 28*28*3， 要先resize成 64*64*3
CHANNELS_IMG = 1

NUM_EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
device = "gpu" if torch.cuda.is_available() else "cpu"


FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)],
            [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

# transforms = transforms.Compose(
#     [
#         transforms.Resize(IMAGE_SIZE),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, ), (0.5, )),
#     ]
# )

#-------------------
# 生成器
#-------------------

class Generator(nn.Module):
    def __init__(self,
                 channels_noise, # 100
                 channels_img, # 64
                 features_g): # features_g = 32。
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            # N x features_g*16 x 4 x 4
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            # N x features_g*16 x 8 x 8
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            # N x features_g*16 x 16 x 16
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            # N x features_g*16 x 32 x 32
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            # N x channels_img x 64 x 64
            nn.ConvTranspose2d(
                features_g * 2,
                channels_img,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self,
               in_channels,
               out_channels,
               kernel_size,
               stride,
               padding):
        return nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                out_channels,
                                                kernel_size,
                                                stride,
                                                padding,
                                                bias=False),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU()
                             )

    def forward(self, x):
        return self.gen(x)

#-------------------
# 判别器
#-------------------
class Discriminator(nn.Module):
    def __init__(self,
                 channel_img, # shaper: N x channels_img x 64 x 64
                 features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            # N x features_d x 32 x 32
            nn.Conv2d(in_channels = channel_img,
                      out_channels = features_d,
                      kernel_size = 4,
                      stride = 2,
                      padding = 1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            # N x features_d*2 x 16 x 16
            self._block(features_d,     features_d * 2, 4, 2, 1),
            # N x features_d*4 x 8 x 8
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            # N x features_d*8 x 4 x 4
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            # N x 1 x 1 x 1
            nn.Conv2d(in_channels = features_d * 8,
                      out_channels = 1,
                      kernel_size = 4,
                      stride = 2,
                      padding = 0),
            nn.Sigmoid()
        )

    def _block(self,
               in_channels,
               out_channels,
               kernel_size,
               stride,
               padding):
        return nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 3 ,64, 64

    noise_dim = 100

    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels,
                         8)
    assert disc(x).shape == (N, 1, 1, 1), "Disariminator test failed"

    gen = Generator(noise_dim,
                    in_channels,
                    8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"

# test()

dataset = Dataset.MNIST(root = "dataset/",
                        train = True,
                        transform = transforms,
                        download=False)

dataloader = DataLoader(dataset,
                        batch_size = BATCH_SIZE,
                        shuffle=True)

gen = Generator(NOISE_DIM, # NOISE_DIM = 256
                CHANNELS_IMG, # CHANNELS_IMG = 1
                FEATURES_GEN).to(device) # FEATURES_GEN = 64

disc = Discriminator(CHANNELS_IMG, # CHANNELS_IMG = 1
                     FEATURES_DISC).to(device) # FEATURES_DISC = 64

initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(),
                     lr=LEARNING_RATE,
                     betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(),
                      lr=LEARNING_RATE,
                      betas=(0.5, 0.999))

criterion = nn.BCELoss()

# 用来观察生成器效果
fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)

writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

print("Starting Training...")

for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    # for batch_idx, (real, _) in enumerate(dataloader):
    for batch_idx, (real, targets) in enumerate(dataloader):
        # ------------------------------------------
        ### 1、训练判别器Train Discriminator:
        #  max log(D(x)) + log(1 - D(G(z)))
        # ------------------------------------------
        real = real.to(device)
        # 拉平成一维
        disc_real = disc(real).reshape(-1)
        # 标签平滑化（让判别器对于真实图片的预测值没有那么confident）
        # 1.1  log(D(real))
        # 标签是1，二元交叉熵函数化简得到这个式子： log(D(real))
        loss_disc_real = criterion(disc_real,
                                   torch.ones_like(disc_real) * 0.9) # 标签

        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        fake = gen(noise)
        # 拉平成一维
        # `fake.detach()`告诉pytorch不要 trace these gradients
        disc_fake = disc(fake.detach()).reshape(-1)
        # 1.2 log(1 - D(G(z)))
        # 标签是0，二元交叉熵函数化简得到这个式子： log(1 - D(G(z)))
        loss_disc_fake = criterion(disc_fake,
                                   torch.zeros_like(disc_fake))

        loss_disc = (loss_disc_real + loss_disc_fake) / 2

        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # ------------------------------------------
        ### 2、训练生成器 Train Generator:
        # min log(1 - D(G(z))) <-> max log(D(G(z))
        # ------------------------------------------

        # 拉平成一维
        output = disc(fake).reshape(-1)

        # 标签为1 ，二元交叉熵化简后得到 max log(D(G(z))
        loss_gen = criterion(output,
                             torch.ones_like(output)) # 标签

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
