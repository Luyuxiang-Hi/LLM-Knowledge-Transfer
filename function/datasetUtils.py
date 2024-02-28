
from PIL import Image
import os
import random
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import functional as TF

from .Tools.get_sam_embedding.utils.transforms import ResizeLongestSide

class SEGData(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=False, get_embedding=False, embedding_type='sam', origin_image_size=256):
        self.data_path = data_path
        id_list_path = os.path.join(self.data_path, 'image_id_list.txt')

        with open(id_list_path, 'r') as file:
            self.id_list = [line.strip() for line in file]

        self.origin_image_size = origin_image_size
        self.transform = transform
        self.embedding_type = embedding_type
        self.get_embedding = get_embedding

        # 定义转换
        if self.get_embedding and embedding_type=='sam':
            self.img_size = 1024
            self.resize1024 = ResizeLongestSide(self.img_size)
            self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
            self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        if self.get_embedding and embedding_type=='dinov2':
            self.img_size = 224
            self.dinov2_norm = T.Normalize((0.420, 0.411, 0.296), (0.213, 0.156, 0.143))
            self.resize224 = ResizeLongestSide(self.img_size)

        # 不能开启transform同时获取embedding
        if self.transform and self.get_embedding:
            raise ValueError 
        # 这里只考虑了256和512两种大小的图片
        if self.origin_image_size!=256 and self.origin_image_size!=512:
            raise ValueError

    def transform_image_label(self, img, label, seed=0):
        random.seed(seed)
        torch.manual_seed(seed)


        # 应用相同的变换到图像和标签
        # 生成随机转换参数
        angle = T.RandomRotation.get_params([-15, 15])
        flip = torch.rand(1) < 0.5

        # 应用转换
        img = TF.rotate(img, angle)
        label = TF.rotate(label, angle)

        if flip:
            img = TF.hflip(img)
            label = TF.hflip(label)

        # 转换回Tensor
        img = TF.to_tensor(img)
        label = TF.to_tensor(label)

        return img, label

    def __getitem__(self, index):

        seed = torch.initial_seed() % (2**32-1) + index

        # 读取图像和标签
        img_path = os.path.join(self.data_path, 'image', self.id_list[index])
        label_path = os.path.join(self.data_path, 'label', self.id_list[index])

        # 读取图像的embedding
        img_embedding = torch.empty(size=(0, 64, 64))
        if self.embedding_type=='sam':
            embedding_path_name = 'sam_embedding'
        elif self.embedding_type=='dinov2':
            embedding_path_name = 'dinov2_embedding'
        if self.get_embedding:
            embedding_path = os.path.join(self.data_path, embedding_path_name, self.id_list[index][:-4]+'.pt')
            img_embedding = torch.load(embedding_path)['img_embedding'] # （256, 64, 64）/ 
        
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            # 提取前三个通道
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r,g,b))
        label = Image.open(label_path)
        if label.mode == 'RGBA':
            # 提取第一个通道
            r, g, b, a = label.split()
            label = Image.merge("L", (r,))
        if label.mode == 'RGB':
            # 提取第一个通道
            r, g, b = label.split()
            label = Image.merge("L", (r,))
        
        # 保证输入的图片为origin_image_size
        if self.origin_image_size==256:
            img = TF.resize(img, [256, 256])
            label = TF.resize(label, [256, 256])
        else:
            img = TF.resize(img, [512, 512])
            label = TF.resize(label, [512, 512])
        
        if self.transform:    
            img, label = self.transform_image_label(img, label, seed=seed)

        elif self.get_embedding and self.embedding_type=='sam':
            # 将数据转为tensor，这一步会做归一化
            # 转为numpy数组，使用meta官方建议的方法resize图像
            img_np = np.array(img).astype(np.uint8)  # (origin_H, origin_W, 3)
            resized_image = self.resize1024.apply_image(img_np)  # (origin_H, origin_W, 3) => (1024, origin_W*scale, 3) / (origin_H*scale, 1024, 3)
            label_np = np.array(label).astype(np.uint8)
            resized_laebl = self.resize1024.apply_image(label_np)
            resized_laebl = np.expand_dims(resized_laebl, axis=-1)

            # 转为tensor，进行归一化，并将图像pad到1024*1024
            image_torch = torch.as_tensor(resized_image)
            image_torch = image_torch.permute(2, 0, 1).contiguous() # (H, W, 3) => (3, H, W)
            image_torch = (image_torch - self.pixel_mean) / self.pixel_std

            label_torch = TF.to_tensor(TF.to_pil_image(resized_laebl))

            # （3， H, W）=> (3, img_size, img_size)
            img = TF.pad(image_torch, (0, self.img_size-image_torch.shape[2], 0, self.img_size-image_torch.shape[1]))
            label = TF.pad(label_torch, (0, self.img_size-label_torch.shape[2], 0, self.img_size-label_torch.shape[1]))

        elif self.get_embedding and self.embedding_type=='dinov2':
            if self.origin_image_size==256:
                img = TF.center_crop(img, [224, 224])
                img = TF.to_tensor(img)
                img = self.dinov2_norm(img)

                label = TF.center_crop(label, [224, 224])
                label = TF.to_tensor(label)
            else:
                img_np = np.array(img).astype(np.uint8)  # (origin_H, origin_W, 3)
                resized_image = self.resize224.apply_image(img_np)  # (origin_H, origin_W, 3) => (224, origin_W*scale, 3) / (origin_H*scale, 224, 3)

                label_np = np.array(label).astype(np.uint8)
                resized_laebl = self.resize224.apply_image(label_np)
                resized_laebl = np.expand_dims(resized_laebl, axis=-1)

                image_torch = TF.to_tensor(TF.to_pil_image(resized_image))
                image_torch = self.dinov2_norm(image_torch)
                
                label_torch = TF.to_tensor(TF.to_pil_image(resized_laebl))

                # (3,  H, W) => (3, img_size, img_size)
                img = TF.pad(image_torch, (0, self.img_size-image_torch.shape[2], 0, self.img_size-image_torch.shape[1]))
                label = TF.pad(label_torch, (0, self.img_size-label_torch.shape[2], 0, self.img_size-label_torch.shape[1]))

        else:
            img = TF.to_tensor(img)
            label = TF.to_tensor(label)

        # 将标签转换为onehot
        label = torch.where(label < 0.5, torch.tensor(0), torch.tensor(1))
        label = F.one_hot(label.long(), 2).permute(0, 3, 1, 2)

        return {
            "img": img.type(torch.FloatTensor), 
            "label": label[0].float(),
            "name": self.id_list[index],
            "img_embedding": img_embedding
            }

    def __len__(self):
        return len(self.id_list)

def getdata(data_path, seed, transform=False, get_embedding=False, embedding_type='sam', origin_image_size=256):
    ds = SEGData(data_path, transform, get_embedding, embedding_type, origin_image_size)
    torch.manual_seed(seed)
    split_length = int(round(0.2 * len(ds)))
    train_length = len(ds) - 2 * split_length
    train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [train_length, split_length, split_length])

    return train_ds, val_ds, test_ds

if __name__ == '__main__':
    data_path = r'../DP_dataset'

    # topil = transforms.ToPILImage()
    # img = topil(single['img'])
    # img.show()

    train_ds, val_ds, _ = getdata(data_path, get_embedding=True)
    print(len(train_ds))
    single = train_ds.__getitem__(0)
    single = train_ds.__getitem__(1)
    single = train_ds.__getitem__(len(train_ds)-1)