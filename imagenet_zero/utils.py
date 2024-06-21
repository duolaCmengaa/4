from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import csv
import numpy as np
import os

# 定义参数
args = {"dataset": 'cifar100', 
      "model": 'resnet18',  
      "batch_size" : 128,
      "epochs" : 100,   
}



def get_train_dataloader(batch_size):
    mean=[x/255.0 for x in [125.3, 123.0, 113.9]]
    std=[x/255.0 for x in [63.0, 62.1, 66.7]]
    # 训练集预处理
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    load_data = os.path.join(script_dir, 'datasets')
    cifar100_train_dataset = datasets.CIFAR100(root=load_data,train=True, transform=train_transform, download=True)
    cifar100_train_loader = DataLoader(dataset=cifar100_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return cifar100_train_loader

def get_test_dataloader():
    mean=[x/255.0 for x in [125.3, 123.0, 113.9]]
    std=[x/255.0 for x in [63.0, 62.1, 66.7]]
    # 测试集预处理
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    load_data = os.path.join(script_dir, 'datasets')
    cifar100_test_dataset = datasets.CIFAR100(root=load_data, train=False, transform=test_transform, download=True)
    cifar100_test_loader = DataLoader(dataset=cifar100_test_dataset, batch_size=args['batch_size'], shuffle=False, pin_memory=True)
    return cifar100_test_loader


class CSVLogger:
    def __init__(self, fieldnames, method ='baseline'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, 'results')
        filename = os.path.join(results_dir, 'CIFAR100_ResNet18_' + method + '.csv')
        self.csv_file = open(filename, 'a')
        writer = csv.writer(self.csv_file)
        writer.writerow([''])
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()
        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()






