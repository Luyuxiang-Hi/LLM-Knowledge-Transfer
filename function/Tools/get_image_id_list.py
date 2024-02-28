import os
# import re
import fnmatch
from tqdm import tqdm


def get_file_list(data_dir, output_txt):
    # 文件组织结构
    # --data
    #   --image
    #     --1.tif
    #     --2.tif
    #     --...
    #   --label
    #     --1.tif
    #     --2.tif
    #     --...
    img_filename_list = os.listdir(os.path.join(data_dir, 'image'))
    img_filename_list = fnmatch.filter(img_filename_list, '*.tif')
    # img_filename_list = sorted(img_filename_list,key=(lambda x: int(re.findall(r"\d+",x)[0])))

    label_filename_list = os.listdir(os.path.join(data_dir, 'label'))
    label_filename_list = fnmatch.filter(img_filename_list, '*.tif')
    # label_filename_list = sorted(img_filename_list,key=(lambda x: int(re.findall(r"\d+",x)[0])))


    img_filename_list = tqdm(img_filename_list)
    with open(output_txt, 'w') as file:
        for i, img_filename in enumerate(img_filename_list):
            if(label_filename_list[i] == img_filename):
                file.write(img_filename + '\n')
            else:
                print(f'error, label and image not match \nimg: {img_filename} \nlabel {label_filename_list[i]}')

if __name__=='__main__':
        
    data_dir = r'./02pv'
    output_txt = r'./02pv/image_id_list.txt'
    get_file_list(data_dir, output_txt)