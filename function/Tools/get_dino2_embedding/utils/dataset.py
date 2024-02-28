import torch
from torchvision.transforms import functional as TF
from torch.nn import functional as F
import torchvision.transforms as T

from PIL import Image
import os
import numpy as np

from .transforms import ResizeLongestSide



class TIFDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, origin_image_size=256):
        self.data_path = data_path
        id_list_path = os.path.join(self.data_path, 'image_id_list.txt')

        self.origin_image_size = origin_image_size

        with open(id_list_path, 'r') as file:
            self.id_list = [line.strip() for line in file]
        
        self.img_size = 224
        self.resize224 = ResizeLongestSide(self.img_size)
        self.dinov2_norm = T.Normalize((0.420, 0.411, 0.296), (0.213, 0.156, 0.143))

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
        
        if self.origin_image_size == 256:
            # 保证输入图片的原始大小为256
            img = TF.resize(img, [256, 256])
            # 中心裁剪至224x224
            img = TF.center_crop(img, [224, 224])
            img = TF.to_tensor(img)
            img = self.dinov2_norm(img)
        
        else:
            img = TF.resize(img, [512, 512])
            img_np = np.array(img).astype(np.uint8)  # (origin_H, origin_W, 3)
            resized_image = self.resize224.apply_image(img_np)  # (origin_H, origin_W, 3) => (224, origin_W*scale, 3) / (origin_H*scale, 224, 3)

            image_torch = TF.to_tensor(TF.to_pil_image(resized_image))
            image_torch = self.dinov2_norm(image_torch)

            # (3,  H, W) => (3, img_size, img_size)
            img = F.pad(image_torch, (0, self.img_size-image_torch.shape[2], 0, self.img_size-image_torch.shape[1]))


        return {
            "img": img.type(torch.FloatTensor), 
            "name": self.id_list[index]
            }

    def __len__(self):
        return len(self.id_list)


if __name__ == '__main__':
    data_path = r'../../../../DP_dataset'

    train_ds  =TIFDataset(data_path)
    single = train_ds.__getitem__(79)