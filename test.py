import datetime
from PIL import Image
from functools import partial
import numpy as np
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from function.datasetUtils import getdata, SEGData
from function.utils import load_checkpoint, batch_to_cuda, seed_everything, worker_init_fn, pretty_print_dictionary       
from function.metrics import compute_batch_metrics


from function.model.unet import Net
# from function.model.unet_res50.unet_res50 import Net
# from function.model.unet_sam import Net
# from function.model.unet_dinov2 import Net
# from function.model.deeplabv3.deeplabv3 import Net
# from function.model.deeplabv3_sam.deeplabv3 import Net
# from function.model.deeplabv3_dinov2.deeplabv3 import Net


def get_args():
    args = {
        "data_path": "../GreenHouseImage/2022.05.17_GH_2m/",
        "log_path": "./export-2m-unet/log",
        "checkpoint_path": "/home/mapbox_select/luyuxiang/HISTORY/export-unet/checkpoint",
        "other_save_path":"./export-2m-unet/other",

        "network":'unet',
        "get_embedding": False,                                     # 只有**_sam或者**_dinov2的网络才需要打开get_embedding
        "embedding_type":'sam',                                     # 可以设置为"sam"、"dinov2"
        "origin_image_size":512,                                     # 输入的图片size，数据集只包含256*256和512*512大小的图片，所以getdata只考虑了这两种情况，这里只能设置为256或者512
        "batchsize": 2,                                            # 这是每张卡的batchsize，最终总的batchsize需要乘以显卡数量
        "num_workers": 4,                                           # 读取数据的线程数量，当GPU读取速度远大于内存读取速度时候开启，设置为0关闭
    }
    return args


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def test(args):

    seed = 666
    seed_everything(seed)

    # 根据args读取数据
    data_path = args["data_path"]
    checkpoint_path = args["checkpoint_path"]
    log_path = args["log_path"]
    other_save_path = args["other_save_path"]

    get_embedding = args["get_embedding"]
    embedding_type = args["embedding_type"]
    origin_image_size = args["origin_image_size"]
    batchsize = args["batchsize"]
    num_workers = args["num_workers"]

    local_rank = int(os.environ["LOCAL_RANK"])
    nprocs = torch.cuda.device_count()
    device = torch.device("cuda", local_rank)

    args['nprocs'] = nprocs
    args['final_activate'] = 'softmax'
    args['loss'] = 'BCE & DICE'
    
    if local_rank == 0:
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H: %M: %S')
        print(f'######################################## time now@{nowtime}  ########################################\nSTART testing')
        formatted_table = pretty_print_dictionary(args)
        print(formatted_table)
    
        # 检查文件夹是否存在，如果没有就创建文件夹
        os.makedirs(other_save_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(other_save_path + '/pred', exist_ok=True)
        os.makedirs(other_save_path + '/label', exist_ok=True)
        with open(os.path.join(log_path, 'args.txt'), 'w') as file:
            file.write(formatted_table)
        
        if not os.path.exists(checkpoint_path):
            raise ValueError

    # 初始化DDP，使用默认backend(nccl)就行。如果是CPU模型运行，需要选择其他后端。
    dist.init_process_group(backend='nccl')
    if local_rank == 0:
        print("Gpu Device Count : ", nprocs)

    # 实例化模型
    model = Net()

    # 将模型放到GPU上, 然后包装模型
    model = model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])


    # 实例化数据对象
    # _, _, test_ds = getdata(data_path, seed, get_embedding=get_embedding, embedding_type=embedding_type, origin_image_size=origin_image_size)
    test_ds = SEGData(data_path, get_embedding=get_embedding, embedding_type=embedding_type, origin_image_size=origin_image_size)
    
    # 定义将数据分配到每一块GPU上的规则
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=batchsize, num_workers = num_workers, pin_memory=True, 
                                                   drop_last = False, sampler=test_sampler, 
                                                   worker_init_fn=partial(worker_init_fn, rank=local_rank, seed=seed))

    # 加载checkpoint
    checkpoint = load_checkpoint(checkpoint_path, local_rank)
    model.module.load_state_dict(checkpoint['model_state_dict'])

    
    #  训练结束之后在验证集上计算各种指标
    model.eval()
    metrics_batch_num = 0
    
    # 在test数据集上跑一下模型
    with torch.no_grad():

        metrics = {"IoU": torch.tensor(0, device=device, dtype=torch.float), "Precision": torch.tensor(0, device=device, dtype=torch.float), \
               "Accuracy": torch.tensor(0, device=device, dtype=torch.float), "Recall": torch.tensor(0, device=device, dtype=torch.float), "F1sorce": torch.tensor(0, device=device, dtype=torch.float)}
        if local_rank == 0:
            pbar = tqdm(total=len(test_dataloader), desc="computing on TEST set...")
        for iteration, batch in enumerate(test_dataloader):
            test_dataloader.sampler.set_epoch(0)
            batch = batch_to_cuda(batch, local_rank)

            # 预测结果
            if get_embedding:
                pred = model(batch['img'], batch['img_embedding'])
            else:
                pred = model(batch['img'])

            # 计算指标
            pred = torch.max(pred, dim=1)[1]   # (B, C, H, W) => (B, H, W)
            label = batch['label'][:, 1, ...]  # (B, C, H, W) => (B, H, W)
            batch_metrics = compute_batch_metrics(label, pred)

            # 累加指标
            for i, key in enumerate(metrics.keys()):
                metrics[key] += batch_metrics[i]
            metrics_batch_num += 1
            
            # 保存一下test预测的图片
            if local_rank == 0:
                for i, img in enumerate(pred):
                    img_np= img.cpu().clone().numpy().astype(np.uint8)*255
                    Image.fromarray(img_np).save(f"{other_save_path}/pred/{batch['name'][i]}.png")
                for i, img in enumerate(label):
                    img_np= img.cpu().clone().numpy().astype(np.uint8)*255
                    Image.fromarray(img_np).save(f"{other_save_path}/label/{batch['name'][i]}.png")
            if local_rank == 0:
                pbar.update(1)
        if local_rank == 0:
            pbar.close()
    
    # 等一下其他的GPU，计算所有GPU上的平均指标
    dist.barrier()
    for key in metrics.keys():
        metrics[key] = reduce_mean(metrics[key], nprocs)
    
    if local_rank == 0:
        with open(os.path.join(log_path, 'metrics.txt'), 'w') as file:
            # 把最后一个epoch的指标储存并打印
            for key in metrics.keys():
                metrics[key] /= metrics_batch_num
                print(f"TESTAverage {key}: {metrics[key]}")
                file.write(f"TESTAverage {key}: {metrics[key]} \n")


if __name__ == '__main__':
    args = get_args()
    test(args)