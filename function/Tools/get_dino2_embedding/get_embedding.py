import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# import pandas as pd

import torch
import torch.distributed as dist
import torchvision.transforms as T

from modeling.SSLVisionTransformer import SSLVisionTransformer
# from modeling.regressor import RNet
from utils.load_checkpoint import load_checkpoint
from utils.dataset import TIFDataset
from utils.batch_to_cuda import batch_to_cuda


def get_args():
    args = {
        "data_path": r'../../../../GreenHouseImage/2022.05.17_GH_2m/',
        "norm_path": r'../../../backboneWeight/aerial_normalization_quantiles_predictor.ckpt',
        "checkpoint": r'../../../backboneWeight/SSLhuge_satellite.pth',
        "save_path": r'../../../../GreenHouseImage/2022.05.17_GH_2m/dinov2_embedding/',

        "origin_image_size": 512, 
        "batchsize": 6,
        "num_workers": 4,
    }
    return args

def get_img_embedding(args):
    data_path = args['data_path']
    norm_path = args["norm_path"]
    checkpoint = args['checkpoint']
    save_path = args['save_path']

    origin_image_size = args["origin_image_size"]
    batchsize = args['batchsize']
    num_workers = args['num_workers']

    os.makedirs(save_path, exist_ok=True)

    nprocs = torch.cuda.device_count()
    if nprocs > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0

    # # 定义模型，用于预处理图像
    # model_norm = RNet(n_classes=6)
    # model_norm = model_norm.eval()

    # # 加载处理图像的checkpoint
    # ckpt = torch.load(norm_path, map_location='cpu')
    # state_dict = ckpt['state_dict']
    # for k in list(state_dict.keys()):
    #     if 'backbone.' in k:
    #         new_k = k.replace('backbone.','')
    #         state_dict[new_k] = state_dict.pop(k)
    # model_norm.load_state_dict(state_dict)
        

    # 定义模型
    image_encoder=SSLVisionTransformer(
            embed_dim=1280,
            num_heads=20,
            out_indices=(9, 16, 22, 29),
            depth=32,
            pretrained=None
        )
    
    # 加载模型权重
    image_encoder = load_checkpoint(image_encoder, checkpoint)

    # 添加虚拟参数，以便于DDP包装
    if nprocs > 1:
        image_encoder.dummy_param = torch.nn.Parameter(torch.empty(0))

    # 模型转移到GPU
    image_encoder.cuda(local_rank)

    # 使DDP封装模型
    if nprocs > 1:
        dist.init_process_group(backend='nccl')
        image_encoder = torch.nn.parallel.DistributedDataParallel(image_encoder, device_ids=[local_rank])
    
    # 模型转为评估模式
    image_encoder.eval()

    # 获取数据
    ds = TIFDataset(data_path, origin_image_size)
    if nprocs > 1:
        data_sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=False)
    else:
        data_sampler = None
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batchsize, num_workers = num_workers, pin_memory=True, 
                                                drop_last = False, sampler=data_sampler)
    if nprocs > 1:
        dataloader.sampler.set_epoch(0)

    # 创建一个空的 DataFrame 用于存储类型
    # df = pd.DataFrame()


    with torch.no_grad():
        # rand_tensor = torch.randn(1, 3, 224, 224).to(local_rank)
        # out = image_encoder(norm(rand_tensor))

        # for i, lst in enumerate(out):
        #     # 获取每个列表中张量的形状
        #     tensor_shapes = [tensor.shape for tensor in lst]
        #     # 添加到 DataFrame
        #     df[f'List {i+1}'] = tensor_shapes
        # print(df)

        for iteration, batch in enumerate(dataloader):

            batch = batch_to_cuda(batch, local_rank)
            # 图像已经归一化，输入的值范围是[0, 1]
            img_embeddings = image_encoder(batch['img'])  # tuple( [ (B, 1280, 14, 14), (B, 1280) ]*4 )

            # region 保存4个tensor
            batch_size = img_embeddings[-1][0].shape[0]
            batch_wise_tensors = [[] for _ in range(batch_size)]
            for b in range(batch_size):
                for list_tensors in img_embeddings:
                    # 只分割第一个Tensor，形状为 (B, 1280, 14, 14)
                    first_tensor = list_tensors[0]
                    # 将同一个Batch的Tensor添加到对应的列表中
                    batch_wise_tensors[b].append(first_tensor[b])
            
            for i, batch_tensors in enumerate(batch_wise_tensors):
                # 创建要保存的字典
                tensor_dict = {f'tensor_{j+1}': tensor.cpu().clone().detach() for j, tensor in enumerate(batch_tensors)}

                # 构造文件名并保存
                file_path = os.path.join(save_path, batch['name'][i][:-4]+'.pt')

                torch.save(tensor_dict, file_path)
            # endregion
            
            # region 保存最后一个tensor
            # for i ,img_embedding in enumerate(img_embeddings[-1][0]):
            #     torch.save({
            #        'img_embedding':img_embedding.cpu().clone().detach(),
            #     }, os.path.join(save_path, batch['name'][i][:-4]+'.pt'))
            #     print(f"save image embedding {batch['name'][i][:-4]}.pt sucessed !")
            # endregion

if __name__ == '__main__':
    args = get_args()
    get_img_embedding(args)
