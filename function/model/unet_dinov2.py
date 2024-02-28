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


class Net(torch.nn.Module):
    def __init__(self, n_channels=3, n_hidden_base=64, num_classes = 2, **kwds):
        super(Net, self ).__init__()

        self.inc = InConv(n_channels, n_hidden_base)
        self.down1 = Down(n_hidden_base, n_hidden_base*2)
        self.down2 = Down(n_hidden_base*2, n_hidden_base*4)
        self.down3 = Down(n_hidden_base*4, n_hidden_base*8)
        self.down4 = Down(n_hidden_base*8, n_hidden_base*16)

        self.up1 = Up(n_hidden_base*16 + 1280, n_hidden_base*8, n_hidden_base*8)
        self.up2 = Up(n_hidden_base*8 + 1280, n_hidden_base*4, n_hidden_base*4)
        self.up3 = Up(n_hidden_base*4 + 1280, n_hidden_base*2, n_hidden_base*2)
        self.up4 = Up(n_hidden_base*2 + 1280, n_hidden_base, n_hidden_base)

        self.seg_module=nn.Sequential(
            nn.Conv2d(n_hidden_base,n_hidden_base,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(n_hidden_base),
            nn.ReLU(),
            nn.Conv2d(n_hidden_base, n_hidden_base, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_hidden_base),
            nn.ReLU(),
            nn.Conv2d(n_hidden_base,num_classes,3,1,1),
            nn.Softmax(dim=num_classes)
        )


    def forward(self, x, embedding):

        # x0 = self.inc.forward(x)     # torch.Size([2, 64, 224, 224])
        # x1 = self.down1.forward(x0)  # torch.Size([2, 128, 112, 112])
        # x2 = self.down2.forward(x1)  # torch.Size([2, 256, 56, 56])
        # x3 = self.down3.forward(x2)  # torch.Size([2, 512, 28, 28])

        # bottom = self.down4.forward(x3)  # torch.Size([2, 1024, 14, 14])
        # bottom = torch.cat([bottom, embedding], dim=1)  # torch.Size([2, 2304, 14, 14])

        # y3 = self.up1.forward(bottom, x3)  # torch.Size([2, 512, 28, 28])
        # y2 = self.up2.forward(y3, x2)  # torch.Size([2, 256, 56, 56])
        # y1 = self.up3.forward(y2, x1)  # torch.Size([2, 128, 112, 112])
        # features = self.up4.forward(y1, x0)  # torch.Size([2, 64, 224, 224])

        # seg = self.seg_module(features)

        # return seg

        x0 = self.inc.forward(x)
        x1 = self.down1.forward(x0)
        x2 = self.down2.forward(x1)
        x3 = self.down3.forward(x2)

        bottom = self.down4.forward(x3)
        bottom = torch.cat([bottom, embedding['tensor_4']], dim=1)

        tensor3 = embedding['tensor_3']
        tensor3 = F.interpolate(tensor3, scale_factor=4, mode='bilinear', align_corners=False) # torch.Size([2, 1280, 56, 56])
        tensor2 = embedding['tensor_2']
        tensor2 = F.interpolate(tensor2, scale_factor=8, mode='bilinear', align_corners=False) # torch.Size([2, 1280, 112, 112])
        tensor1 = embedding['tensor_1']
        tensor1 = F.interpolate(tensor1, scale_factor=16, mode='bilinear', align_corners=False) # torch.Size([2, 1280, 224, 224])
        y3 = self.up1.forward(bottom, x3)
        y2 = self.up2.forward(y3, torch.cat([x2, tensor3], dim=1))
        y1 = self.up3.forward(y2, torch.cat([x1, tensor2], dim=1))
        features = self.up4.forward(y1, torch.cat([x0, tensor1], dim=1))

        seg = self.seg_module(features)

        return seg


if __name__ == '__main__':

    model = Net()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: ", total_params)
    # model = Net().cuda()
    # x = torch.randn(2, 3, 224, 224).cuda()

    # embedding = {
    #     'tensor_1':torch.randn(2, 1280, 14, 14).cuda(),
    #     'tensor_2':torch.randn(2, 1280, 14, 14).cuda(),
    #     'tensor_3':torch.randn(2, 1280, 14, 14).cuda(),
    #     'tensor_4':torch.randn(2, 1280, 14, 14).cuda(),
    # }
    # seg = model(x, embedding)
    # print(seg.size())
