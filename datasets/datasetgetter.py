import torch
from torchvision.datasets import ImageFolder
import os
import torchvision.transforms as transforms
from datasets.custom_dataset import ImageFolerRemap, ImageFolerRemapSingle

class Compose(object):
    def __init__(self, tf):
        self.tf = tf

    def __call__(self, img):
        for t in self.tf:
            img = t(img)
        return img


def get_dataset(args):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    normalize = transforms.Normalize(mean=mean, std=std)

    transform = Compose([transforms.Resize((args.img_size, args.img_size)),
                                   transforms.ToTensor(),
                                   normalize])

    transform_val = Compose([transforms.Resize((args.img_size, args.img_size)),
                                       transforms.ToTensor(),
                                       normalize])

    class_to_use = args.att_to_use#[0,1]

    print('USE CLASSES', class_to_use)

    # remap labels
    remap_table = {}
    i = 0
    for k in class_to_use:
        remap_table[k] = i
        i += 1

    print("LABEL MAP:", remap_table) #{0: 0, 1: 1}


    img_dir = args.data_dir #"./data"
#图片读取

    #数据集读取
    dataset = ImageFolerRemapSingle(img_dir, transform=transform, remap_table=remap_table)#包括了三个部分('./data/font1/1547.png', 1, './data/font0/1547.png')
    valdataset = ImageFolerRemap(img_dir, transform=transform_val, remap_table=remap_table)

# parse classes to use
    tot_targets = torch.tensor(dataset.targets)#tensor([0, 0, 0,  ..., 1, 1, 1])
    min_data = 99999999
    max_data = 0

    train_idx = None
    val_idx = None

    for k in class_to_use:
        tmp_idx = torch.nonzero(tot_targets == k, as_tuple=False)  # 找出对应字体类型的所有字#1-1549//1550-3099
        train_tmp_idx = tmp_idx[:-args.val_num]  # 划分训练集和测试集1-1499
        val_tmp_idx = tmp_idx[-args.val_num:]
        if k == class_to_use[0]:#font0是内容图片
            train_tmp_idx_class_1 = train_tmp_idx
            train_idx = train_tmp_idx.clone()
            val_idx = val_tmp_idx.clone()
        else:   #shifengge
            train_idx = torch.cat((train_idx, train_tmp_idx))  # 将所有坐标连接所有的训练集
            val_idx = torch.cat((val_idx, val_tmp_idx))#所有的测试集
            train_tmp_idx_class_2 = train_tmp_idx# 这里有点懵逼
        if min_data > len(train_tmp_idx):
            min_data = len(train_tmp_idx)
        if max_data < len(train_tmp_idx):
            max_data = len(train_tmp_idx)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(valdataset, val_idx)
    # print('class1',train_tmp_idx_class_1,type(train_tmp_idx_class_1),train_tmp_idx_class_1.size(0))
    # index = torch.randperm(0, train_tmp_idx_class_1.size(0), 1)
    # print(index,type(index))

    train_dataset_class_1 = torch.utils.data.Subset(dataset, train_tmp_idx_class_1)
    train_dataset_class_2 = torch.utils.data.Subset(dataset, train_tmp_idx_class_2)#风格  获取指定一个索引序列对应的子数据集。


    args.min_data = min_data
    args.max_data = max_data
    print("MINIMUM DATA :", args.min_data)
    print("MAXIMUM DATA :", args.max_data)

    train_dataset = {'TRAIN': train_dataset, 'FULL': dataset}

    return train_dataset_class_1 #包括了三个部分('./data/font1/1547.png', 1, './data/font0/1547.png'),
#train_dataset_class_1 第一类字体，train_dataset_class_2第二类字体     凤：应该修改第二类字体
#if __name__ == "__main__":


def get_dataset1(args):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    normalize = transforms.Normalize(mean=mean, std=std)

    transform = Compose([
                                   transforms.ToTensor(),
                                   normalize])

    transform_val = Compose([
                                       transforms.ToTensor(),
                                       normalize])

    class_to_use = args.att_to_use#[0,1]

    print('USE CLASSES', class_to_use)

    # remap labels
    remap_table = {}
    i = 0
    for k in class_to_use:
        remap_table[k] = i
        i += 1

    print("LABEL MAP:", remap_table) #{0: 0, 1: 1}

#换成匹配数据集的路径
    img_dir = "./test0416/Yik/data1" #"./data"
#图片读取

    #数据集读取
    dataset = ImageFolerRemapSingle(img_dir, transform=transform, remap_table=remap_table)#包括了三个部分('./data/font1/1547.png', 1, './data/font0/1547.png')
    valdataset = ImageFolerRemap(img_dir, transform=transform_val, remap_table=remap_table)

# parse classes to use
    tot_targets = torch.tensor(dataset.targets)#tensor([0, 0, 0,  ..., 1, 1, 1])
    tot_targets = torch.tensor(dataset.targets)#tensor([0, 0, 0,  ..., 1, 1, 1])
    min_data = 99999999
    max_data = 0

    train_idx = None
    val_idx = None

    for k in class_to_use:
        tmp_idx = torch.nonzero(tot_targets == k, as_tuple=False)  # 找出对应字体类型的所有字#1-1549//1550-3099
        train_tmp_idx = tmp_idx[:-args.val_num]  # 划分训练集和测试集1-1499
        val_tmp_idx = tmp_idx[-args.val_num:]
        if k == class_to_use[0]:#font0是内容图片
            train_tmp_idx_class_1 = train_tmp_idx
            train_idx = train_tmp_idx.clone()
            val_idx = val_tmp_idx.clone()
        else:   #shifengge
            train_idx = torch.cat((train_idx, train_tmp_idx))  # 将所有坐标连接所有的训练集
            val_idx = torch.cat((val_idx, val_tmp_idx))#所有的测试集
            train_tmp_idx_class_2 = train_tmp_idx# 这里有点懵逼
        if min_data > len(train_tmp_idx):
            min_data = len(train_tmp_idx)
        if max_data < len(train_tmp_idx):
            max_data = len(train_tmp_idx)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(valdataset, val_idx)

    # print('class1',train_tmp_idx_class_1,type(train_tmp_idx_class_1),train_tmp_idx_class_1.size(0))
    # index = torch.randperm(0, train_tmp_idx_class_1.size(0), 1)
    # print(index,type(index))

    train_dataset_class_1 = torch.utils.data.Subset(dataset, train_tmp_idx_class_1)
    train_dataset_class_2 = torch.utils.data.Subset(dataset, train_tmp_idx_class_2)#风格  获取指定一个索引序列对应的子数据集。


    args.min_data = min_data
    args.max_data = max_data
    print("MINIMUM DATA :", args.min_data)
    print("MAXIMUM DATA :", args.max_data)

    train_dataset = {'TRAIN': train_dataset, 'FULL': dataset}

    return train_dataset, val_dataset,train_dataset_class_2 #包括了三个部分('./data/font1/1547.png', 1, './data/font0/1547.png'),


