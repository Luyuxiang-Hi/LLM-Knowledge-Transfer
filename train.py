import math
import datetime
from PIL import Image
from functools import partial
import numpy as np
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from torch.utils.tensorboard import SummaryWriter
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from function.datasetUtils import getdata
from function.loss_fun.loss import SegLoss
from function.utils import load_checkpoint, batch_to_cuda, seed_everything, weights_init, worker_init_fn, get_lr, pretty_print_dictionary       
from function.metrics import compute_batch_metrics

# from function.model.unet_sam import Net
# from function.model.unet_dinov2 import Net
# from function.model.unet import Net
# from function.model.unet_res50.unet_res50 import Net
# from function.model.deeplabv3.deeplabv3 import Net
# from function.model.deeplabv3_sam.deeplabv3 import Net
# from function.model.deeplabv3_dinov2.deeplabv3 import Net

def get_args():
    args = {
        "data_path": "../GreenHouseImage/DP_dataset/",              # 数据集，应该包含/image 和/label，如果get_embedding=True，还应该包括embedding文件
        "log_path": "./export/log",                                 # 输出日志、loss、预测的评价指标
        "checkpoint_path": "./export/checkpoint",                   # 保存的checkpoint
        "other_save_path":"./export/other",                         # 保存预测的图片
        "backboneWeight":"./backboneWeight/resnet50-19c8e357.pth",  # 只有unet-res50需要加载backbone
        "batchsize": 32,                                             # 这是每张卡的batchsize，最终总的batchsize需要乘以显卡数量
        "epochs": 200,                                              # 一共要跑这么多个epoch
        "Freeze_epochs":50,                                          # 前Freeze_epochs个epoch会冻结backbone，为0时候不冻结
        "init_lr": 2e-3,                                            # 初始学习率
        "lr_multiple_init":1,                                       # 初始学习率倍率，即epoch=0时候的学习率倍率，实际学习率 = 初始学习率*学习率倍率
        "lr_multiple_final":0.01,                                   # 最终学习率倍率，即epoch=epochs时候的学习率倍率
        "load_cp": False,                                           # 是否需要断点续传（加载上次中断的训练）
        "num_workers": 4,                                           # 读取数据的线程数量，当GPU读取速度远大于内存读取速度时候开启，设置为0关闭
        "num_classes": 2,                                           # 分类数量
        "transform": False,                                          # 是否开启数据增强，不能与get_embedding同时为Ture
        "get_embedding":True,                                       # 是否获取图像的embedding，开启时transform必须关闭
        "save_step": 5,                                             # 每save_step个epoch保存一次checkpoint
        "pretrained": True,                                         # 是否加载backbone的预训练权重
        "model_init_type": 'normal',                                # 模型初始化方法，可选参数normal，xavier， kaiming，orthogonal
        "embedding_type": 'dinov2'                                     # 使用哪种模型的embedding，可选参数sam， dinov2
    }
    return args


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def train(args):

    seed = 666
    seed_everything(seed)

    # 根据args读取数据
    data_path = args["data_path"]
    checkpoint_path = args["checkpoint_path"]
    log_path = args["log_path"]
    other_save_path = args["other_save_path"]
    backboneWeight = args["backboneWeight"]
    batchsize = args["batchsize"]
    epochs = args["epochs"]
    Freeze_epochs = args["Freeze_epochs"]
    init_lr = args["init_lr"]
    lr_multiple_init = args["lr_multiple_init"]
    lr_multiple_final = args["lr_multiple_final"]
    load_cp = args["load_cp"]
    num_classes = args['num_classes']
    transform = args["transform"]
    get_embedding = args["get_embedding"]
    embedding_type = args["embedding_type"]
    num_workers = args["num_workers"]
    save_step = args["save_step"]
    pretrained = args["pretrained"]
    model_init_type = args["model_init_type"]

    local_rank = int(os.environ["LOCAL_RANK"])
    nprocs = torch.cuda.device_count()
    device = torch.device("cuda", local_rank)

    args['nprocs'] = nprocs
    args['final_activate'] = 'softmax'
    args['loss'] = 'BCE & DICE'
    
    if local_rank == 0:
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H: %M: %S')
        print(f'######################################## time now@{nowtime}  ########################################\nSTART training')
        formatted_table = pretty_print_dictionary(args)
        print(formatted_table)
    
        # 检查文件夹是否存在，如果没有就创建文件夹
        os.makedirs(other_save_path, exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        with open(os.path.join(log_path, 'args.txt'), 'w') as file:
            file.write(formatted_table)

    # 初始化DDP，使用默认backend(nccl)就行。如果是CPU模型运行，需要选择其他后端。
    dist.init_process_group(backend='nccl')
    if local_rank == 0:
        print("Gpu Device Count : ", nprocs)

    # 实例化模型
    model = Net(num_classes = num_classes, pretrained = pretrained, checkpoint = backboneWeight)

    # 显卡数量大于1时候同步所有显卡的BN
    if nprocs>1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # 不加载backbone的预训练权重就初始化模型
    if not pretrained:
        weights_init(model, model_init_type)
    
    # 将模型放到GPU上
    model = model.cuda(local_rank)

    # 如果冻结训练，在这里冻结backbone，设置find_unused_parameters=True
    if Freeze_epochs != 0:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        model.module.freeze_backbone()
    else: 
        model = DDP(model, device_ids=[local_rank])

    # 实例化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    # 定义学习率衰退方案,x为自动传入的当前epoch，计算出的结果乘以初始lr就是当前epoch的lr
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (lr_multiple_init - lr_multiple_final) + lr_multiple_final
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 定义损失函数
    loss_fun = SegLoss().to(local_rank)

    # 实例化数据对象
    train_ds, val_ds, test_ds = getdata(data_path, seed, transform=transform, get_embedding=get_embedding, embedding_type=embedding_type)
    
    # 定义将数据分配到每一块GPU上的规则
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True)
    vaild_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds, shuffle=False)
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batchsize, num_workers = num_workers, pin_memory=True, 
                                                   drop_last = True, sampler=train_sampler, 
                                                   worker_init_fn=partial(worker_init_fn, rank=local_rank, seed=seed))
    vaild_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=batchsize, num_workers = num_workers, pin_memory=True, 
                                                   drop_last = True, sampler=vaild_sampler, 
                                                   worker_init_fn=partial(worker_init_fn, rank=local_rank, seed=seed))
    test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=batchsize, num_workers = num_workers, pin_memory=True, 
                                                   drop_last = False, sampler=test_sampler, 
                                                   worker_init_fn=partial(worker_init_fn, rank=local_rank, seed=seed))

    # 只在第一个进程中实例化tensorboard
    tb_writer = None
    if dist.get_rank() == 0:
        tb_writer = SummaryWriter(log_dir=log_path)

    # 加载checkpoint
    start_epoch = 0
    if load_cp==True:
        checkpoint = load_checkpoint(checkpoint_path, local_rank)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    # 开始训练
    Freeze_flag = True
    for epoch in range(start_epoch, epochs):
        if local_rank ==0 :
            start_time = datetime.datetime.now()
        model.train()
        if Freeze_epochs !=0 and epoch >= Freeze_epochs and Freeze_flag:
            model.module.unfreeze_backbone()
            Freeze_flag = False
        train_dataloader.sampler.set_epoch(epoch)
        vaild_dataloader.sampler.set_epoch(epoch)

        # 初始化每个epoch的loss
        train_loss = 0
        sum = 0
        
        ################### 开始对数据遍历 #######################
        for i, batch in enumerate(train_dataloader):

            # 复制数据到gpu上
            batch = batch_to_cuda(batch, local_rank)

            # 预测结果
            if get_embedding:
                pred = model(batch['img'], batch['img_embedding'])
            else:
                pred = model(batch['img'])

            # 计算损失函数并回传梯度
            loss = loss_fun(pred, batch['label'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算每个batch中gt和pre的平均loss
            train_loss += loss
            sum += 1
        ##################### 数据遍历完成 ########################

        # 每次epoch衰减学习率
        lr_scheduler.step()

        # 每次epoch验证
        with torch.no_grad():
            vaild_loss, vaild_metrics = vaild(model, vaild_dataloader, epoch, local_rank, loss_fun, get_embedding, device)

         # 等待所有进程到此
        torch.distributed.barrier()

        # 计算所有显卡上loss的平均值
        average_train_loss = reduce_mean(train_loss/sum , nprocs)
        average_vaild_loss = reduce_mean(vaild_loss, nprocs)
        if epoch % 5 == 0:
            for key in vaild_metrics.keys():
                vaild_metrics[key] = reduce_mean(vaild_metrics[key], nprocs)
 
        if local_rank == 0:
            if epoch % 5 ==0:
                for key in vaild_metrics.keys():
                    tb_writer.add_scalar(f'Val/{key}', vaild_metrics[key], epoch+1)
            # 保存loss
            tb_writer.add_scalar('train_loss', average_train_loss, epoch + 1)
            tb_writer.add_scalar('val_loss', average_vaild_loss, epoch + 1)

            # 保存每次epoch的checkpoint
            if (epoch + 1) % save_step == 0 or epoch + 1 == epochs:
              torch.save({
                 'epoch': epoch + 1,
                 'model_state_dict': model.module.state_dict(),
                 'optimizer_state_dict':optimizer.state_dict(),
                 'lr_scheduler_state_dict':lr_scheduler.state_dict()
              }, os.path.join(checkpoint_path, "checkpoint.epoch_{:04d}.tar".format(epoch+1)))

            # 打印
            epoch_time = (datetime.datetime.now() - start_time).total_seconds()
            print(f"epoch: {epoch+1}/{epochs}_________DOWN_________"
                  f"@TrainLoss: {average_train_loss:.4f}  "
                  f"@VaildLoss: {average_vaild_loss:.4f}  "
                  f"@LearningRate: {get_lr(optimizer):.7f}  @Tiem: {epoch_time:.4f} s ")
    #  训练结束
    
    #  训练结束之后在验证集上计算各种指标
    model.eval()
    metrics_batch_num = 0
    
    # 新建两个文件夹保存最后一个epoch预测图片
    if local_rank == 0:
        os.makedirs(other_save_path + '/pred', exist_ok=True)
        os.makedirs(other_save_path + '/label', exist_ok=True)
    
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
            pred = torch.max(pred, dim=1)[1]
            label = batch['label'][:, 1, ...]
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
                tb_writer.add_scalar(f'Test/{key}', metrics[key], 1)
                print(f"TESTAverage {key}: {metrics[key]}")
                file.write(f"TESTAverage {key}: {metrics[key]} \n")
        # 关闭tensorboard
        tb_writer.close()


def vaild(model, dataloader, epoch, local_rank, loss_fun, get_embedding, device):
    model.eval()
    val_loss = 0
    sum = 0
    metrics = None
    if epoch % 5 == 0:
        metrics = {"IoU": torch.tensor(0, device=device, dtype=torch.float), "Precision": torch.tensor(0, device=device, dtype=torch.float), \
               "Accuracy": torch.tensor(0, device=device, dtype=torch.float), "Recall": torch.tensor(0, device=device, dtype=torch.float), "F1sorce": torch.tensor(0, device=device, dtype=torch.float)}
    for i, batch in  enumerate(dataloader):

        # 复制数据到gpu上
        batch = batch_to_cuda(batch, local_rank)

        # 预测结果
        if get_embedding:
            pred = model(batch['img'], batch['img_embedding'])
        else:
            pred = model(batch['img'])

        # 计算loss 
        loss = loss_fun(pred, batch['label'])

        # 计数，计算总的loss
        val_loss += loss
        sum += 1

        if epoch % 5 ==0:
            # 计算指标
            pred = torch.max(pred, dim=1)[1]
            label = batch['label'][:, 1, ...]
            batch_metrics = compute_batch_metrics(label, pred)
            # 累加指标
            for i, key in enumerate(metrics.keys()):
                metrics[key] += batch_metrics[i]
    if epoch % 5 ==0:
        for key in metrics.keys():
            metrics[key] /= sum
    
    return val_loss/sum, metrics


if __name__ == '__main__':
    args = get_args()
    train(args)