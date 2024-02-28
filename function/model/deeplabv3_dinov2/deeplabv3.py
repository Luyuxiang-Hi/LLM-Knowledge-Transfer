from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabV3
from .resnet import resnet50
# import torch


def Net(num_classes=2, output_stride=16, pretrained=False, checkpoint=''):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet50(
        pretrained=pretrained,
        checkpoint = checkpoint,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048 + 1280

    # return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    # low_level_planes = 256
    # classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)

    return_layers = {'layer4': 'out'}
    classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

# if __name__ == '__main__':
#     net = Net().cuda()
#     tensor  = torch.randn(2, 3, 224, 224).cuda()
#     embedding = {'tensor_1':torch.randn(2, 1280, 14, 14).cuda(),
#                  'tensor_2':torch.randn(2, 1280, 14, 14).cuda(),
#                  'tensor_3':torch.randn(2, 1280, 14, 14).cuda(),
#                  'tensor_4':torch.randn(2, 1280, 14, 14).cuda(),
#                  }
#     out = net(tensor, embedding)