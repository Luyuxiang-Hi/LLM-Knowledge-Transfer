from functools import partial
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
import torch
import torch.distributed as dist

from modeling.image_encoder import ImageEncoderViT
from utils.load_checkpoint import load_checkpoint
from utils.dataset import TIFDataset
from utils.batch_to_cuda import batch_to_cuda


def get_args():
    args = {
        "data_path": r'../../../../GreenHouseImage/2022.10.24_05DP/',
        "checkpoint": r'../../../backboneWeight/sam_vit_h_4b8939.pth',
        "save_path": r'../../../../GreenHouseImage/2022.10.24_05DP/sam_embedding/',
        "encoder_embed_dim": 1280,                         # vit_b: 768, vit_h: 1024
        "encoder_depth": 32,                              # vit_b: 12, vit_h: 32
        "encoder_num_heads": 16,                          # vit_b: 12, vit_h: 16
        "encoder_global_attn_indexes": [7, 15, 23, 31],     # vit_b: [2, 5, 8, 11], vit_h: [7, 15, 23, 31]

        "origin_image_size": 256,                           # 256 or 512 only
        "batchsize": 6,                                     # 32Gcuda: image224->6  image 512 -> 
        "num_workers": 4,
    }
    return args

def get_img_embedding(args):
    data_path = args['data_path']
    checkpoint = args['checkpoint']
    save_path = args['save_path']
    encoder_embed_dim = args['encoder_embed_dim']
    encoder_depth = args['encoder_depth']
    encoder_num_heads = args['encoder_num_heads']
    encoder_global_attn_indexes = args['encoder_global_attn_indexes']
    origin_image_size = args["origin_image_size"]
    batchsize = args['batchsize']
    num_workers = args['num_workers']

    os.makedirs(save_path, exist_ok=True)
    local_rank = int(os.environ["LOCAL_RANK"])
    nprocs = torch.cuda.device_count()

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
    image_encoder.cuda(local_rank)

    # 使DDP封装模型
    if nprocs > 1:
        dist.init_process_group(backend='nccl')
        image_encoder = torch.nn.parallel.DistributedDataParallel(image_encoder, device_ids=[local_rank])
    
    # 模型转为评估模式
    image_encoder.eval()

    # 获取数据
    ds = TIFDataset(data_path, origin_image_size=origin_image_size)
    if nprocs > 1:
        data_sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=False)
    else:
        data_sampler = None
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batchsize, num_workers = num_workers, pin_memory=True, 
                                                drop_last = False, sampler=data_sampler)
    if nprocs > 1:
        dataloader.sampler.set_epoch(0)
    with torch.no_grad():
        for iteration, batch in enumerate(dataloader):

            batch = batch_to_cuda(batch, local_rank)
            img_embeddings = image_encoder(batch['img'])  # (B, 256, 64, 64)

            for i ,img_embedding in enumerate(img_embeddings):
                torch.save({
                   'img_embedding':img_embedding.cpu().clone().detach(),
                }, os.path.join(save_path, batch['name'][i][:-4]+'.pt'))
                print(f"save image embedding {batch['name'][i][:-4]}.pt sucessed !")

if __name__ == '__main__':
    args = get_args()
    get_img_embedding(args)
