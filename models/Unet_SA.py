import torch
import torch.nn as nn
from .spectral import SpectralNorm
import torch.nn.functional as fct


def normalize(x_in):
    x_out = x_in.clone()
    for i in range(x_in.size(0)):
        for j in range(x_in.size(1)):
            x_max = torch.max(x_in[i, j, :, :])
            x_min = torch.min(x_in[i, j, :, :])
            x_out[i, j, :, :] = (x_out[i, j, :, :] - x_min) / (x_max - x_min)

    return x_out


class SelfAttn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super().__init__()
        self.chanel_in = in_dim
        self.activation = activation
        ##  下面的query_conv，key_conv，value_conv即对应Wg,Wf,Wh
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=max(1, in_dim // 8), kernel_size=1)  # 即得到C^ X C
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=max(1, in_dim // 8), kernel_size=1)  # 即得到C^ X C
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # 即得到C X C
        self.gamma = nn.Parameter(torch.zeros(1))  # 这里即是计算最终输出的时候的伽马值，初始化为0

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        #  下面的proj_query，proj_key都是C^ X C X C X N= C^ X N
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X (N) X C,permute为转置
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check，进行点乘操作
        attention = self.softmax(energy)  # B X (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class ResNet(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.model = nn.Sequential(
            SpectralNorm(nn.Conv2d(channels, channels, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),

            SpectralNorm(nn.Conv2d(channels, channels, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, inputs):
        return self.model(inputs)


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)


class DoubleConv(nn.Module):
    """(convolution => [IN] => LeakyReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(0.1, inplace=True),

            SpectralNorm(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downscaling = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.downscaling(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        # input is CHW
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = fct.pad(x2, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class Generator(nn.Module):
    """Generator."""

    def __init__(self, batch_size=64, image_size=64, z_dim=1, conv_dim=64):
        super().__init__()
        self.batch_size = batch_size
        self.imsize = image_size
        self.out_channel = 3
        self.unet_layer = 5

        self.d1 = DoubleConv(z_dim, conv_dim)
        curr_dim = conv_dim  # 1*64*128 → 64*64*128
        self.d2 = Down(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2  # 128*32*64
        self.d3 = Down(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2  # 256*16*32
        self.d4 = Down(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2  # 512*8*16
        self.d5 = Down(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2  # 1024*4*8

        self.u4 = Up(curr_dim, curr_dim // 2)
        curr_dim = curr_dim // 2  # 512*8*16
        self.u3 = Up(curr_dim, curr_dim // 2)
        curr_dim = curr_dim // 2  # 256*16*32
        self.u2 = Up(curr_dim, curr_dim // 2)
        curr_dim = curr_dim // 2  # 128*32*64
        self.u1 = Up(curr_dim, curr_dim // 2)
        curr_dim = curr_dim // 2  # 64*64*128

        self.res = ResNet(curr_dim)
        self.jump = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim, kernel_size=1),
            nn.InstanceNorm2d(curr_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.attn1 = SelfAttn(512, 'relu')
        self.attn2 = SelfAttn(1024, 'relu')
        self.last = nn.Conv2d(curr_dim, self.out_channel, 3, 1, 1)  # 3*64*128

    def unet(self, x):
        x1 = self.d1(x)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        x4, _ = self.attn1(x4)
        x5 = self.d5(x4)
        x5, _ = self.attn2(x5)
        x = self.u4(x4, x5)
        x = self.u3(x3, x)
        x = self.u2(x2, x)
        x = self.u1(x1, x)
        return x

    def forward(self, x):
        x = self.unet(x)
        for i in range(12):
            res_out = self.res(x)
            x = x + self.jump(res_out)

        x = self.last(x)
        x = normalize(x)
        return x


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64):
        super().__init__()
        self.imsize = image_size
        self.batch_size = batch_size
        self.in_dim = 3
        self.out_dim = 1

        self.position = nn.Sequential(
            nn.Conv2d(self.in_dim, conv_dim, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(conv_dim, conv_dim // 2, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(conv_dim // 2, self.out_dim, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            View([self.batch_size, 8 * 16]),
            nn.Linear(8 * 16, 10),
            View([self.batch_size, 5, 2]))

        self.layer0 = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.in_dim, conv_dim, kernel_size=1)),
            nn.InstanceNorm2d(conv_dim),
            nn.LeakyReLU(0.1, inplace=True),
            SpectralNorm(nn.Conv2d(conv_dim, conv_dim, kernel_size=1)),
            nn.InstanceNorm2d(conv_dim),
            nn.LeakyReLU(0.1, inplace=True))
        curr_dim = conv_dim  # 3*64*128 → 64*64*128

        self.layer1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(curr_dim, curr_dim, 4, 2, 1)),
            nn.LeakyReLU(0.1, inplace=True))
        curr_dim = conv_dim  # 64*64*128 → 64*32*64

        self.layer2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)),
            nn.LeakyReLU(0.1, inplace=True))
        curr_dim = curr_dim * 2  # 128*16*32

        self.layer3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)),
            nn.LeakyReLU(0.1, inplace=True))
        curr_dim = curr_dim * 2  # 256*8*16

        self.layer4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)),
            nn.LeakyReLU(0.1, inplace=True))
        curr_dim = curr_dim * 2  # 512*4*8

        self.last = nn.Sequential(nn.Conv2d(curr_dim, self.out_dim, (4, 8)))

        self.attn1 = SelfAttn(256, 'relu')
        self.attn2 = SelfAttn(512, 'relu')

    def forward(self, x):
        pred_layer = []
        out = self.layer0(x)
        pred_layer.append(out)

        out = self.layer1(out)
        pred_layer.append(out)

        out = self.layer2(out)
        pred_layer.append(out)

        out = self.layer3(out)
        out, _ = self.attn1(out)
        pred_layer.append(out)

        out = self.layer4(out)
        out, _ = self.attn2(out)
        pred_layer.append(out)

        pred_bool = self.last(out)
        pred_position = self.position(x)

        return pred_bool.squeeze(), pred_layer, pred_position
