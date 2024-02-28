import torch.nn.functional as F
import torch.nn as nn
import torch


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch, no_padding=False):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=0 if no_padding else 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=0 if no_padding else 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ELU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, no_padding=False):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch, no_padding)

    def forward(self, x):
        x = self.conv.forward(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, no_padding=False):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, no_padding)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch_1, in_ch_2, out_ch, no_padding=False):
        super(Up, self).__init__()
        self.conv = DoubleConv(in_ch_1 + in_ch_2, out_ch, no_padding)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv.forward(x)
        return x

# new_size = (old + 2p -k ) / s + 1  常用的保持形状不变的有 k,s,p = 1, 1, 0   k,s,p = 3, 1, 1
class Net(torch.nn.Module):
    def __init__(self, n_channels=3, n_hidden_base=64, num_classes = 2, **kwds):
        super(Net, self ).__init__()

        self.inc = InConv(n_channels, n_hidden_base)
        self.down1 = Down(n_hidden_base, n_hidden_base*2)
        self.down2 = Down(n_hidden_base*2, n_hidden_base*4)
        self.down3 = Down(n_hidden_base*4, n_hidden_base*8)
        self.down4 = Down(n_hidden_base*8, n_hidden_base*16)

        self.up1 = Up(n_hidden_base*16, n_hidden_base*8, n_hidden_base*8)
        self.up2 = Up(n_hidden_base*8, n_hidden_base*4, n_hidden_base*4)
        self.up3 = Up(n_hidden_base*4, n_hidden_base*2, n_hidden_base*2)
        self.up4 = Up(n_hidden_base*2, n_hidden_base, n_hidden_base)

        self.seg_module=nn.Sequential(
            nn.Conv2d(n_hidden_base, num_classes, kernel_size=1,stride=1,padding=0),
            nn.Softmax(dim=num_classes)
        )


    def forward(self, x):

        x0 = self.inc.forward(x)
        x1 = self.down1.forward(x0)
        x2 = self.down2.forward(x1)
        x3 = self.down3.forward(x2)
        y4 = self.down4.forward(x3)
        y3 = self.up1.forward(y4, x3)
        y2 = self.up2.forward(y3, x2)
        y1 = self.up3.forward(y2, x1)
        features = self.up4.forward(y1, x0)

        seg = self.seg_module(features)

        return seg


if __name__ == '__main__':
    model = Net().cuda()
    x = torch.randn(20, 3, 512, 512).cuda()
    seg = model(x)
    print(seg.size())
