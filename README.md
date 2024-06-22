# 对比监督学习和自监督学习在图像分类任务上的性能表现

基本要求：

(1) 实现SimCLR自监督学习算法并使用该算法在CIFAR-10数据集上训练ResNet-18，随后在CIFAR-100数据集中使用Linear Classification Protocol对其性能进行评测；

(2) 将上述结果与在ImageNet数据集上采用监督学习训练得到的表征在相同的协议下进行对比，并比较二者相对于在CIFAR-100数据集上从零开始以监督学习方式进行训练所带来的提升；

(3) 尝试不同的超参数组合，探索自监督预训练数据集规模对性能的影响；



## 准备
首先下载好本仓库的所有文件，并配置好环境

```
git clone https://github.com/duolaCmengaa/4.git
```

<details>
  <summary> Experimental environment (click to expand) </summary>
  
  ## Experimental environment
  - pytorch == 2.3.0
  - pytorch-cuda == 12.1
  - torchvision == 0.18.0

  - matplotlib == 3.6.2

  - tensorboard == 2.10.1
  - tqdm == 4.66.2
  - opencv-python == 4.10.0.82
  - numpy == 1.24.3

</details>



## 文件存放路径

因为程序会自动下载数据集，所以不需要手动下载
模型权重和Tensorboard日志可以前往[MineResult](https://drive.google.com/drive/folders/1-DZqKcJj7YhARVyFru8imsI3Y5ez0YHE)下载，各个文件夹里所要用到的文件如下所示

### imagenet_zero

```
                                                                                      
├── resnet_18.py
├── test1.py
├── train1.py
├── utils.py
├── results
│   └── final
│   │   └──CIFAR100_ResNet18_baseline.csv "最终的日志文件"
│   │   └──zero_CIFAR100_ResNet18.pth "最终的模型权重"
│   │   └──finallog 
│   │   │   └──events.out.tfevents.1718137586.c32264eda658.2848610.6 "最终的日志文件"
│   └── train_zero
│   │   └──events.out.tfevents.1718154991.c32264eda658.2848610.11 "训练过程产生的日志文件，不是最终的日志文件"
│   └── pictures
│   │   └──lr_batch_zero_train.png 
│   └── CIFAR100_ResNet18_baseline.csv "训练过程产生的日志文件，不是最终的日志文件"

```

### simclr_imagenet

```
                                                                                      
├── gaussian_blur.py "高斯模糊"
├── image_lin.py
├── imagenet_test.py
├── models.py
├── simclr.py
├── simclr1.py
├── simclr_config.yaml
├── simclr_lin.py
├── simclr_test.py
├── logs
│   └── SimCLR
│   │   └── cifar10
│   │   │   └── image_lin.log "记录了运行的一些日志"
│   │   │   └── simclr.log
│   │   │   └── simclr_lin.log
│   │   │   └── .hydra
│   │   │   │   └── config.yaml
│   │   │   │   └── hydra.yaml
│   │   │   │   └── overrides.yaml
├── result
│   └── ImageNet_train "SimCLR训练得到的模型在CIFAR-100数据集中使用Linear Classification Protocol"
│   │   └── log
│   │   │   └── events.out.tfevents.1718946241.c32264eda658.718905.0
│   │   └── models
│   │   │   └── imagenet_best.pth
│   └── Linear_Classification_Protocol "在ImageNet数据集上采用监督学习训练得到的模型使用Linear Classification Protocol"
│   │   └── log
│   │   │   └── events.out.tfevents.1718941797.c32264eda658.705964.0
│   │   └── models
│   │   │   └── simclr_lin_resnet18_best.pth
│   └── train "SimCLR在cifar10训练"
│   │   └── log
│   │   │   └── events.out.tfevents.1718030418.c32264eda658.2666795.0
│   │   └── models
│   │   │   └── simclr_resnet18_epoch500.pt
│   │   │   └── simclr_resnet18_epoch1000.pt
│   │   │   └── simclr_resnet18_epoch1500.pt
│   │   │   └── simclr_resnet18_epoch2000.pt

```

### 实现SimCLR自监督学习算法并使用该算法在CIFAR-10数据集上训练ResNet-18
首先

```
cd simclr_imagenet
```

#### 开始训练

如果只用cifar10数据集的训练集运行训练则运行

```
python simclr.py
```

如果用cifar10数据集的训练集和测试集一起训练则运行

```
python simclr1.py
```

如果要修改模型参数和训练参数直接去修改simclr_config.yaml文件内的参数就好


#### 使用Linear Classification Protocol对其性能进行评测

#####  对SimCLR训练好的模型进行评测

前往[MineResult](https://drive.google.com/drive/folders/1-DZqKcJj7YhARVyFru8imsI3Y5ez0YHE)下载train_and_test或者only_train内的result文件夹，其中only_train是只用cifar10训练集训练得到的结果，而train_and_test是用cifar10的训练集和测试集一起训练得到的结果，将result文件夹
放置在正确的路径，之后运行

```
python simclr_lin.py
```

即可正常运行，如果要修改训练epoch直接前往simclr_config.yaml修改finetune_epochs参数就好，最后会在"result/Linear_Classification_Protocol"得到日志和最终的模型权重

#####  对ImageNet数据集上采用监督学习训练得到的表征进行评测

运行
```
python image_lin.py
```

即可正常运行，最后会在"result/ImageNet_train"得到日志和最终的模型权重，同理，如果要修改训练epoch直接前往simclr_config.yaml修改finetune_epochs参数就好


#### 对训练好的模型进行测试

下载并将result文件夹放置在正确的路径，注意如果运行前面的评测部分会覆盖下载好的模型权重，之后运行

```
python simclr_test.py
```

即可得到对SimCLR训练好的模型进行Linear Classification Protocol评测后得到的模型的top1准确率和top5准确率

运行

```
python imagenet_test.py
```

即可得到在ImageNet数据集上采用监督学习训练得到的表征进行Linear Classification Protocol评测后得到的模型的top1准确率和top5准确率




### 在CIFAR-100数据集上从零开始以监督学习方式进行训练

首先

```
cd imagenet_zero
```

#### 开始训练
运行
```
python train1.py
```

如果要指定学习率和batch size前往train1.py修改batch_sizes和learning_rates两个参数。

#### 对模型进行测试
前往[MineResult](https://drive.google.com/drive/folders/1-DZqKcJj7YhARVyFru8imsI3Y5ez0YHE)下载imagenet_zero内的results文件夹，并放置在正确的路径，之后

运行
```
python test1.py
```

可以得到模型的top1准确率和top5准确率


### Tensorboard可视化
运行
```
tensorboard --logdir=path
```
将path替换为日志文件所在的目录路径即可可视化



