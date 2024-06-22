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
  - imageio == 2.34.1
  - imageio-ffmpeg == 0.5.1
  - matplotlib == 3.7.5
  - configargparse == 1.7
  - tensorboard == 2.14.0
  - tqdm == 4.66.4
  - opencv-python == 4.10.0.84 
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

#####  对SimCLR训练好的模型

前往[MineResult](https://drive.google.com/drive/folders/1-DZqKcJj7YhARVyFru8imsI3Y5ez0YHE)下载train_and_test或者only_train内的result文件夹，其中only_train是只用cifar10训练集训练得到的结果，而train_and_test是用cifar10的训练集和测试集一起训练得到的结果，将result文件夹
放置在正确的路径，之后运行

```
python simclr_lin.py
```

即可正常




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





                                                                     
│   ├── nerf_llff_data                                                                                                  
│   │   └── fern/llfftest                                                                                                                             
│   │   │   └──images   
│   │   │   │   └──img01.jpg
│   │   │   │   └──img02.jpg
│   │   │   │   └── ...
│   │   │   └──images_4 "四倍下采样"
│   │   │   │   └──img01.jpg
│   │   │   │   └──img02.jpg
│   │   │   │   └── ...
│   │   │   └──images_8 "八倍下采样"
│   │   │   │   └──img01.jpg
│   │   │   │   └──img02.jpg
│   │   │   │   └── ...
│   │   │   └──sparse
│   │   │   │   └──0
│   │   │   │   │   └──cameras.bin
│   │   │   │   │   └──images.bin
│   │   │   │   │   └──points3D.bin
│   │   │   │   │   └──project.ini
│   │   │   └──database.db 
│   │   │   └──poses_bounds.npy
│   │   │   └──view_imgs.txt "所使用的图片的记录"  
```

### logs

```
                                                                                      
├── logs                                                                                                                                                                                               
│   ├── fern_test/llfftest 
│   │   └── testset_200000 "测试用的图片"
│   │   │   └──img01.jpg
│   │   │   └──img02.jpg
│   │   └── 050000.tar
│   │   └── 100000.tar
│   │   └── 150000.tar
│   │   └── 200000.tar "迭代200000步的模型权重
│   │   └── args.txt
│   │   └── config.txt
│   │   └── llfftest_spiral_200000_rgb.mp4 "最终渲染出的视频"
│   ├── summaries                                                                                                
│   │   └── fern_test/llfftest
│   │   │   │   └── events.out.tfevents.1718818883.c32264eda658.369162.0 "Tensorboard日志文件"                                                                                                                        

```

#### 全局文件存放路径

```
                                                                                      
├── configs                                                                                                                                                                                    │   └── fern.txt
│   └── llfftest.txt       
├── data
│   └── ...
├── logs "当需要从预训练的模型开始训练或者对预留的测试图片进行测试才需要此文件，否则请不要放置此文件"
│   └── ...
├── images_8.py "八倍下采样"
├── load_blender.py
├── load_LINEMOD.py
├── load_deepvoxels.py
├── load_llff.py
├── names.py "批量命名图片"
├── requirements.txt
├── run_nerf.py
├── run_nerf_helpers.py

```

## 训练

将所有所需要的文件按照正确的路径放置后，即可开始训练，从零开始训练不要放logs文件夹

### 如果要从零开始训练fern数据集

放好fern文件夹内的data文件夹后运行

```
python run_nerf.py --config configs/fern.txt
```

如果要从预训练好的模型权重开始继续训练只需将logs文件放置在正确路径，运行相同的命令即可从迭代步数最大的模型权重继续开始训练，训练前记得修改run_nerf.py内的N_iters参数

渲染的视频文件将被保存至logs文件内，如果要修改batch size，只需前往configs文件夹内的对应的txt(fern数据集对应fern.txt，llfftest数据集对应llfftest.txt)文件修改N_rand即可

### 如果要从零开始训练我们的数据集

放好bicycle文件夹内的data文件夹后运行

因为是360°旋转，所以命令稍有不同

```
python run_nerf.py --config configs/llfftest.txt --spherify --no_ndc
```

### 如果要基于训练好的NeRF渲染环绕物体的视频

注意要放好对应的logs文件

对于fern

```
python run_nerf.py --config configs/fern.txt --render_only
```

之后会在"logs/fern_test/renderonly_path_199999"生成渲染后的视频

对于我们的数据集
```
python run_nerf.py --config configs/llfftest.txt --spherify --no_ndc --render_only
```

之后会在"logs/llfftest/renderonly_path_199999"生成渲染后的视频


### 如果要基于训练好的NeRF在预留的测试图片上评价定量结果

注意要放好testdata的data文件夹和logs文件夹，还需要前往configs文件夹内的llfftest.txt文件修改llffhold为1


```
python run_nerf.py --config configs/llfftest.txt --spherify --no_ndc --render_only --render_test
```

之后会打印出测试图片的平均PSNR值并且在"logs/llfftest/renderonly_test_199999"文件内生成测试图片的渲染图像
