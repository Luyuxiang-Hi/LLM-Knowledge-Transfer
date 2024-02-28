import numpy as np
import fnmatch
import os
import re
import torch
import random

def label_to_onehot(label, class_list):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    onehot_map = []
    for class_value in class_list:
        equality = np.equal(label, class_value)
        class_map = np.all(equality, axis=-1)
        onehot_map.append(class_map)
    onehot_map = np.stack(onehot_map, axis=-1).astype(np.float32)
    return onehot_map
    

def load_checkpoint(checkpoint_path, gpu):
    namelist = fnmatch.filter(os.listdir(checkpoint_path), "*.tar")
    if len(namelist) == 0:
                return None

    namelist = sorted(namelist, key=(lambda x: int(re.findall(r"\d+",x)[0])))
    name = namelist[-1]  # Last checkpoint
    filepath = os.path.join(checkpoint_path, name)
    
    if gpu == 0:
        print(f'Load checkpoint {name} from {filepath}')
    checkpoint = torch.load(filepath, map_location="cuda:{}".format(gpu))  # map_location is used to load on current device
    return checkpoint


def batch_to_cuda(batch, gpu):
    # Send data to computing device:
    for key, item in batch.items():
        if hasattr(item, "cuda"):
            batch[key] = item.cuda(gpu) # non_blocking=False
    return batch


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


#---------------------------------------------------#
#   
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def pretty_print_dictionary(dictionary):
    # Find the longest key and value for formatting
    max_key_length = max(len(key) for key in dictionary)
    max_value_length = max(len(str(value)) for value in dictionary.values())

    # Top border of the table
    table = "+" + "-" * (max_key_length + 2) + "+" + "-" * (max_value_length + 2) + "+\n"

    # Rows of the table
    for key, value in dictionary.items():
        key_padding = " " * (max_key_length - len(key))
        value_padding = " " * (max_value_length - len(str(value)))
        table += f"| {key}{key_padding} | {value}{value_padding} |\n"

    # Bottom border of the table
    table += "+" + "-" * (max_key_length + 2) + "+" + "-" * (max_value_length + 2) + "+"

    return table