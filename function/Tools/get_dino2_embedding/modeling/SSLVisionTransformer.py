# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
import math
import warnings

import torch
from torch import nn
from torch.nn.modules.utils import _pair as to_2tuple
import torch.nn.functional as F

from .DinoVisionTransformer import DinoVisionTransformer


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=False):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
                  
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class AdaptivePadding(nn.Module):
    """Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.
    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1.
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):

        super(AdaptivePadding, self).__init__()

        assert padding in ('same', 'corner')

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2
                ])
        return x


class SSLVisionTransformer(DinoVisionTransformer):
    """Vision Transformer.
    """

    def __init__(self,
                interpolate_mode='bicubic',
                init_cfg=None,
                pretrained=None,
                out_indices=(4, 11, 17, 23),
                final_norm=False,
                with_cls_token=True,
                output_cls_token=True,
                frozen_stages=100,
                 *args, **kwargs):
        super(SSLVisionTransformer, self).__init__(*args, **kwargs) 
       
        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

            
        if len(self.blocks)==1:    
            self.blocks = self.blocks[0] 
        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = len(self.blocks) - 1
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of int, list or tuple')

        self.interpolate_mode = interpolate_mode
        self.pretrained = pretrained
        self.frozen_stages = frozen_stages
        self.detach = True
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.final_norm = final_norm
        self.patch_size = self.patch_embed.patch_size
        self.adapad = AdaptivePadding(kernel_size=self.patch_size, stride=self.patch_size, padding='same')
        # if pretrained:
        #     self.init_weights_ssl(pretrained)
        
        self._freeze_stages()

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        """Resize pos_embed weights.
        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed
    
    # def init_weights_ssl(self, pretrained):
    #     print("init_weights", pretrained)
    #     if (isinstance(self.init_cfg, dict)
    #             and self.init_cfg.get('type') == 'Pretrained'):
            
    #         checkpoint = torch.load(pretrained, map_location='cpu')
    #         if 'state_dict' in checkpoint:
    #             # timm checkpoint
    #             state_dict = checkpoint['state_dict']
    #         elif 'model' in checkpoint:
    #             # deit checkpoint
    #             state_dict = checkpoint['model']
    #         elif 'teacher' in checkpoint:
    #             # dino eval checkpoint
    #             state_dict = checkpoint['teacher']
    #         else:
    #             state_dict = checkpoint
            
    #         if len([k for k in state_dict.keys() if 'teacher.backbone.' in k]) > 0:
    #             state_dict = {k.replace('teacher.backbone.', ''):v for k,v in state_dict.items() if 'teacher.backbone' in k}
    #         if len([k for k in state_dict.keys() if 'backbone.' in k]) > 0:
    #             state_dict = {k.replace('backbone.', ''):v for k,v in state_dict.items()}

    #         if 'pos_embed' in state_dict.keys():
    #             if self.pos_embed.shape != state_dict['pos_embed'].shape:
    #                 print(f'Resize the pos_embed shape from '
    #                             f'{state_dict["pos_embed"].shape} to '
    #                             f'{self.pos_embed.shape}')
    #                 h, w = (224, 224) # self.img_size
    #                 pos_size = int(
    #                     math.sqrt(state_dict['pos_embed'].shape[1] - 1))
    #                 state_dict['pos_embed'] = self.resize_pos_embed(
    #                     state_dict['pos_embed'],
    #                     (h // self.patch_size[0], w // self.patch_size[1]),
    #                     (pos_size, pos_size), self.interpolate_mode)
    #         self.load_state_dict(state_dict)
    #     else:
    #         super(SSLVisionTransformer, self).init_weights()
            

    def forward(self, x):
        
        with torch.set_grad_enabled(not self.detach):
            _, _, old_w, old_h = x.shape
            xx = self.adapad(x)
            
            x = F.pad(x, (0, xx.shape[-1] - x.shape[-1], 0, xx.shape[-2] - x.shape[-2]))
            B, nc, w, h = x.shape

            x, _, _ = self.prepare_tokens(x)
            # we return the output tokens from the `n` last blocks
            outs = []
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                if i in self.out_indices:
                    if self.with_cls_token:
                        out = x[:, 1:]
                    else:
                        out = x
                    B, _, C = out.shape
                    out = out.reshape(B, w // self.patch_size[0], h // self.patch_size[1],
                                    C).permute(0, 3, 1, 2).contiguous()
                    if self.output_cls_token:
                        out = [out, x[:, 0]]
                    else:
                        out = [out]
                    if self.final_norm:
                        out = [self.norm(o) for o in out]
                    if self.detach:
                        out = [o.detach() for o in out]
                    outs.append(out)
            return tuple(outs)
    
    def _freeze_stages(self):
        """Freeze stages param and norm stats."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for m in [self.patch_embed]:
                for param in m.parameters():
                    param.requires_grad = False
            self.cls_token.requires_grad = False
            self.pos_embed.requires_grad = False
            self.mask_token.requires_grad = False

        if self.frozen_stages >= len(self.blocks) - 1:
            self.norm.eval()
            for param in self.norm.parameters():
                param.requires_grad = False
            self.detach = True

        for i, layer in enumerate(self.blocks):
            if i <= self.frozen_stages:
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False

                    
