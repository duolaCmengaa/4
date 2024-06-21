import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from utils import args, CSVLogger, get_train_dataloader
from test1 import testing
from torch.optim.lr_scheduler import StepLR
import shutil
import matplotlib.pyplot as plt #plt 用于显示图片
from resnet_18 import ResNet18

def save_best_log(src_log_dir, dest_log_dir):
    if os.path.exists(dest_log_dir):
        shutil.rmtree(dest_log_dir)
    shutil.copytree(src_log_dir, dest_log_dir)

def copy_file(source, destination):
        # 复制文件
        shutil.copy(source, destination)
        print("文件复制成功！")


def train1(epoch,model,criterion,optimizer):
    print('\nEpoch: %d' % epoch)
    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))
        
        if args['cuda']:
            images = images.cuda()
            labels = labels.cuda()

        model.zero_grad()
        pred = model(images)

        xentropy_loss = criterion(pred, labels)
        xentropy_loss.backward()
        optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()

        # 计算训练过程中的准确率
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        # 打印训练过程中的loss和acc
        progress_bar.set_postfix(xentropy='%.3f' % (xentropy_loss_avg / (i + 1)), acc='%.3f' % accuracy)

    return (xentropy_loss_avg / (i + 1)), accuracy



if __name__ == "__main__":
    args['cuda'] = torch.cuda.is_available()

    cudnn.benchmark = True

    torch.manual_seed(0)
    if args['cuda']:
        torch.cuda.manual_seed(0)

    batch_sizes = [64,128,256]
    learning_rates = [0.08,0.06,0.04,0.02]
    best_batch = 0
    best_lr = 0 

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    final_dir = os.path.join(results_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    final2_dir = os.path.join(final_dir, 'finallog')
    os.makedirs(final2_dir, exist_ok=True)
    picture_save_path = os.path.join(results_dir, 'pictures')
    os.makedirs(picture_save_path, exist_ok=True)
    picture_save_path = os.path.join(picture_save_path, 'lr_batch_zero_train.png')
    model1_dir = os.path.join(final_dir, 'zero_CIFAR100_ResNet18.pth')
    glo_acc = 0

    for batch in batch_sizes:
        train_loader = get_train_dataloader(batch)
        best_acc_list = []
        for learning_rate in learning_rates: 
            log_dir = os.path.join(results_dir, 'train_zero')
            filename = os.path.join(results_dir, 'CIFAR100_ResNet18_' + 'baseline' + '.csv')

            tmp=0 # 记录是否为最好成绩

            # 模型
            model = ResNet18(num_classes=100)
            # 定义优化器
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=5e-4)
            # 定义学习率优化
            scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.2)

            if args['cuda']:
                model = model.cuda()
            # 定义损失函数
            if args['cuda']:
                criterion = nn.CrossEntropyLoss().cuda()
            else:
                criterion = nn.CrossEntropyLoss()

            # 数据储存到csv文件
            try:
                os.makedirs(results_dir)
            except:
                pass


            # 训练模型过程
            train = train1

            # 初始化 SummaryWriter
            if os.path.exists(log_dir):
                    shutil.rmtree(log_dir)

            # 检查文件是否存在，然后删除
            if os.path.exists(filename):
                os.remove(filename)
            else:
                pass

            writer = SummaryWriter(log_dir)
            csv_logger = CSVLogger(fieldnames=['epoch', 'train_loss', 'train_acc', 'test_acc'],method='baseline')

            patience = 10  # 设定早停的耐心值
            count = 0
            loc_acc = 0

            for epoch in range(1, args['epochs'] + 1):
                train_loss, train_acc = train(epoch, model, criterion, optimizer)
                test_acc = testing(model)
                tqdm.write('test_acc: %.3f' % test_acc)
                scheduler.step()
                row = {'epoch': str(epoch), 'train_loss':str(train_loss), 'train_acc': str(train_acc), 'test_acc': str(test_acc)}
                csv_logger.writerow(row)
                writer.add_scalar('train_loss', train_loss, global_step=epoch)
                writer.add_scalar('test_acc', test_acc, global_step=epoch)

                # 保存准确率最高的模型
                if test_acc > loc_acc:
                    loc_acc = test_acc
                    count = 0

                    if test_acc > glo_acc:
                        glo_acc = test_acc
                        best_lr = learning_rate
                        best_batch = batch
                        tmp=1
                        torch.save(model.state_dict(), model1_dir)
                    else:
                        pass

                else:
                    count += 1
                    if count >= patience:
                        print("Early stopping at epoch:", epoch)
                        break
            if tmp ==1:
                save_best_log(log_dir, final2_dir)
                copy_file(filename,final_dir)
            writer.close()
            csv_logger.close()

            best_acc_list.append(loc_acc)
        x = range(len(learning_rates))
        plt.plot(x, best_acc_list, marker='o', label=f'batch size: {batch}')

    plt.xlabel('learning rate')
    plt.ylabel('the accuracy of validation_dataset')
    plt.xticks(x, learning_rates)
    # 添加标题
    plt.title('no_pre_train batch sizes and learning rates')
    # 添加图例
    plt.legend()
    plt.savefig(picture_save_path)
    plt.show()


    print(f"对learning_rate和batch size进行搜索,找到最好的模型是learning_rate为{best_lr},batch size为{best_batch}时此时验证集最高的准确率为{glo_acc}")