import torch
import torch.nn as nn

from .resnet import resnet50


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Net(nn.Module):
    def __init__(self, num_classes = 2, pretrained = False, checkpoint = ''):
        super(Net, self).__init__()
       
        self.resnet = resnet50(pretrained = pretrained, checkpoint=checkpoint)
        in_filters  = [192, 512, 1024, 3072]
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor = 2), 
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
            nn.ReLU(),
        )

        self.final = nn.Sequential(nn.Conv2d(out_filters[0], num_classes, 1),
                                   nn.Softmax(dim=num_classes))
        # self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):

        [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        for param in self.resnet.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.resnet.parameters():
                param.requires_grad = True