import os

# 文件夹路径
folder_tif = '../../../GreenHouseImage/DP_dataset/image'
folder_pt = '../../../GreenHouseImage/DP_dataset/dinov2_embedding'

def list_files(folder_path, file_extension):
    """
    列出指定文件夹中特定扩展名的所有文件。
    """
    return [file for file in os.listdir(folder_path) if file.endswith(file_extension)]

def extract_file_names(file_list):
    """
    从文件列表中提取文件名（不含扩展名）。
    """
    return {os.path.splitext(file)[0] for file in file_list}

# 列出两个文件夹中的文件
files_tif = list_files(folder_tif, '.tif')
files_pt = list_files(folder_pt, '.pt')

# 提取文件名（不包含扩展名）
names_tif = extract_file_names(files_tif)
names_pt = extract_file_names(files_pt)

# 计算文件数量和匹配的文件数量
num_files_tif = len(files_tif)
num_files_pt = len(files_pt)
matched_files = names_tif.intersection(names_pt)
num_matched_files = len(matched_files)

# 找出不匹配的文件名
unmatched_tif = names_tif - names_pt
unmatched_pt = names_pt - names_tif

# 输出结果
print("Number of .tif files:", num_files_tif)
print("Number of .pt files:", num_files_pt)
print("Number of matched files:", num_matched_files)
print("Unmatched .tif files:", unmatched_tif)
print("Unmatched .pt files:", unmatched_pt)
