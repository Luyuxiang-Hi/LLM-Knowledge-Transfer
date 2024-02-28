from functools import partial
import os
import torch
from torchvision.transforms import functional as TF
from PIL import Image


from modeling.image_encoder import ImageEncoderViT
from utils.load_checkpoint import load_checkpoint


def get_args():
    args = {
        "img_path": r'../../../../GreenHouseImage/2022.10.24_05DP/',
        "checkpoint": r'D:\GH\train2.0\backboneWeight/sam_vit_b_01ec64.pth',

        "encoder_embed_dim": 1280,                                                         # vit_b: 768, vit_h: 1024
        "encoder_depth": 32,                                                               # vit_b: 12, vit_h: 32
        "encoder_num_heads": 16,                                                           # vit_b: 12, vit_h: 16
        "encoder_global_attn_indexes": [7, 15, 23, 31],                                    # vit_b: [2, 5, 8, 11], vit_h: [7, 15, 23, 31]
    }
    return args

def get_img_embedding(args):

    encoder_embed_dim=768
    encoder_depth=12
    encoder_num_heads=12
    encoder_global_attn_indexes=[2, 5, 8, 11]

    checkpoint = args['checkpoint']

    # encoder_embed_dim = args['encoder_embed_dim']
    # encoder_depth = args['encoder_depth']
    # encoder_num_heads = args['encoder_num_heads']
    # encoder_global_attn_indexes = args['encoder_global_attn_indexes']


    # 定义模型
    image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=256,
        )
    
    # 加载模型权重
    image_encoder = load_checkpoint(image_encoder, checkpoint)

    # 模型转移到GPU
    image_encoder.cuda(0)
    
    # 模型转为评估模式
    image_encoder.eval()

    # 获取数据
    img = torch.randn(1, 3, 256, 256).to(0)

    # 预测
    with torch.no_grad():
        img_embeddings = image_encoder(img)  # (B, 256, 64, 64)
        print(img_embeddings.shape)

if __name__ == '__main__':
    args = get_args()
    get_img_embedding(args)
