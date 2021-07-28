import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard

class Discriminator(nn.Module):
    def __init__(self,
                 img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)
# 超参数
device= "gpu" if torch.cuda.is_available() else "cpu"
z_dim = 100
img_dim = 28 *  28 * 1
learning_rate = 3e-4
batch_size = 32
num_epoches = 50

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
)

dataset = datasets.MNIST(root="dataset/",
                         transform=transforms,
                         download=False)
                         # download=True)

loader = DataLoader(dataset,
                    batch_size=batch_size,
                    shuffle=True)

disc = Discriminator(img_dim).to(device)
gen = Generator(z_dim,
                img_dim).to(device)
# 假数据。用来在tensorboard里观察
fixed_noise = torch.randn((batch_size, z_dim)).to(device)


optim_disc = optim.Adam(disc.parameters(), lr = learning_rate)
optim_gen = optim.Adam(gen.parameters(), lr = learning_rate)

criterion = nn.BCELoss()

writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")

step = 0

for epoch in range(num_epoches):
    # for batch_idx, (image, labels) in enumerate(loader):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # ----------------------------------------------
        # 1、对于判别器：max log(D(real)) + log(1 - D(G(z)))
        # 解释：
        # 1.我们希望D(real) = 1, 又因为 log(1) = 0， 对数函数的自变量取值在(0,1)内，log函数值是负数。所以我们希望 maxinize log(D(real))
        # 2.我们希望D(G(z)) = 0。D(G(z))的范围是(0,1)，所以(1 - D(G(z)))=1.取值范围是(0, 1)，这个取值范围，log函数值是负数。所以我们希望 maxinize log(1 - D(G(z)))
        # ----------------------------------------------
        noise = torch.randn(batch_size, z_dim).to(device)

        fake = gen(noise)

        # disc_real = disc(real.detach()).view(-1)
        disc_real = disc(real).view(-1)

        # 1.1  log(D(real))
        # 标签是1，二元交叉熵函数化简得到这个式子： log(D(real))
        lossD_real = criterion(disc_real,
                               torch.ones_like(disc_real))

        disc_fake = disc(fake).view(-1)

        # 1.2 log(1 - D(G(z)))
        # 标签是0，二元交叉熵函数化简得到这个式子： log(1 - D(G(z)))
        lossD_fake = criterion(disc_fake,
                               torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake) / 2

        disc.zero_grad()

        # lossD.backward()
        lossD.backward(retain_graph=True)

        optim_disc.step()

        # ----------------------------------------------
        # 2、对于生成器 min log (1 - D(G(z)))
        #解释：对于生成器，希望D(G(z))=1, 所以，1 - D(G(z))=0，log值负无穷，即我们希望 minimize log (1 - D(G(z)))
        # 但是，从计算角度考虑，因为这个log (1 - D(G(z)))是 weak gradent。所以换一个等价的式子来计算
        #    min log(1 - D(G(z)))  <-->  max log(D(G(z))
        # -----------------------------------------------
        # 所以要重用上面的 `fake = gen(noise)`
        # 所以不会清除 intermodiate computation中间的计算。让它还在计算图里面
        # 方法一、用detach() : `disc_real = disc(real.detach()).view(-1)`
        # 方法二、lossD.backward(retain_graph=True)，

        disc_fake_for_lossG = disc(fake).view(-1)

        # 标签为1 ，二元交叉熵化简后得到 max log(D(G(z))
        lossG = criterion(disc_fake_for_lossG,
                          torch.ones_like(disc_fake_for_lossG))

        gen.zero_grad()
        lossG.backward()
        optim_gen.step()

        if batch_idx == 0:
            print(f"Epoch[{epoch}/{num_epoches}] \ "
                  f"Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
            )

        with torch.no_grad():
            fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
            data = real.reshape(-1, 1, 28, 28)
            img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
            img_grid_real = torchvision.utils.make_grid(data, normalize=True)

            writer_fake.add_image(
                "Mnist Fake Images", img_grid_fake, global_step=step
            )
            writer_real.add_image(
                "Mnist Real Images", img_grid_real, global_step=step
            )
            step += 1





























