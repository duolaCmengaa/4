# 对比监督学习和自监督学习在图像分类任务上的性能表现

基本要求：

(1) 实现任一自监督学习算法并使用该算法在自选的数据集上训练ResNet-18，随后在CIFAR-100数据集中使用Linear Classification Protocol对其性能进行评测；

(2) 将上述结果与在ImageNet数据集上采用监督学习训练得到的表征在相同的协议下进行对比，并比较二者相对于在CIFAR-100数据集上从零开始以监督学习方式进行训练所带来的提升；

(3) 尝试不同的超参数组合，探索自监督预训练数据集规模对性能的影响；



## 准备
首先下载好本仓库的所有文件，并配置好环境

```
[git clone https://github.com/duolaCmengaa/5.git](https://github.com/duolaCmengaa/4.git)
cd 4

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
我们提供了三个数据集，数据集可以前往[MineData](https://drive.google.com/drive/folders/1JcWnWqofrvRd6TWf9Ix0hpnNLA0eoJkg)下载，fern，bicycle和testdata文件夹里存放了logs和data文件夹，data文件夹保存了数据集，logs文件夹保存了训练好的模型权重，日志文件和渲染的视频，各个文件夹里所要用到的文件如下所示

### data

```
                                                                                      
├── data                                                                                                                                                                                               
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
