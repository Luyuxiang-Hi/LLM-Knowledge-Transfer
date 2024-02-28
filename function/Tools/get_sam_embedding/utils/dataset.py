import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from PIL import Image
import os
import numpy as np

from .transforms import ResizeLongestSide

class TIFDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, origin_image_size=256, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375]):
        self.data_path = data_path
        id_list_path = os.path.join(self.data_path, 'image_id_list.txt')

        with open(id_list_path, 'r') as file:
            self.id_list = [line.strip() for line in file]
        
        self.img_size = 1024
        self.resize = ResizeLongestSide(self.img_size)
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)

        self.origin_image_size = origin_image_size

        # 这里只考虑了256和512两种大小的图片
        if self.origin_image_size!=256 and self.origin_image_size!=512:
            raise ValueError

    def __getitem__(self, index):

        # 读取图像
        img_path = os.path.join(self.data_path, 'image', self.id_list[index])
        img = Image.open(img_path)

        # 如果是RGBA图像，转为RGB
        if img.mode == 'RGBA':
            # 提取第一个通道
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r,g,b))
        
        # 保证输入图片的原始大小为origin_image_size
        if self.origin_image_size == 256:
            img = TF.resize(img, [256, 256])
        else:
            img = TF.resize(img, [512, 512])

        # 转为numpy数组，使用meta官方建议的方法resize图像
        img_np = np.array(img).astype(np.uint8)  # (origin_H, origin_W, 3)
        resized_image = self.resize.apply_image(img_np)  # (origin_H, origin_W, 3) => (1024, origin_W*scale, 3) / (origin_H*scale, 1024, 3)

        # 转为tensor，进行归一化，并将图像pad到1024*1024
        image_torch = torch.as_tensor(resized_image)
        image_torch = image_torch.permute(2, 0, 1).contiguous() # (H, W, 3) => (3, H, W)
        image_torch = (image_torch - self.pixel_mean) / self.pixel_std
        
        # （3， H, W）=> (3, img_size, img_size)
        padding_img = F.pad(image_torch, (0, self.img_size-image_torch.shape[2], 0, self.img_size-image_torch.shape[1]))

        return {
            "img": padding_img, 
            "name": self.id_list[index]
            }

    def __len__(self):
        return len(self.id_list)


if __name__ == '__main__':
    data_path = r'../../../../DP_dataset'

    train_ds  =TIFDataset(data_path)
    single = train_ds.__getitem__(79)