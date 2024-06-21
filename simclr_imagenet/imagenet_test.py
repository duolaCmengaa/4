import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms
from models import SimCLR
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
import os



class LinModel(nn.Module):
    """Linear wrapper of encoder."""
    def __init__(self, encoder: nn.Module, feature_dim: int, n_classes: int):
        super().__init__()
        self.enc = encoder
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.lin = nn.Linear(self.feature_dim, self.n_classes)

    def forward(self, x):
        return self.lin(self.enc(x))

def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate(model, dataloader):
    model.eval()
    top1_acc_meter = AverageMeter('top1_acc')
    top5_acc_meter = AverageMeter('top5_acc')

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            logits = model(x)
            top1_acc, top5_acc = accuracy(logits, y)
            top1_acc_meter.update(top1_acc.item(), x.size(0))
            top5_acc_meter.update(top5_acc.item(), x.size(0))
    
    return top1_acc_meter.avg, top5_acc_meter.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'datasets')
    imagenet_dir = os.path.join(script_dir, 'result', 'ImageNet_train')

    test_set = CIFAR100(root=data_dir, train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    n_classes = 100

    # Prepare model
    resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Identity()
    model = LinModel(resnet18, num_ftrs, n_classes=100)
    model = model.cuda()

    # Load the best saved model
    tmp_path = os.path.join(imagenet_dir, 'models')    
    model_path = os.path.join(tmp_path, 'imagenet_best.pth')
    model.load_state_dict(torch.load(model_path))

    # Evaluate the model
    top1_acc, top5_acc = evaluate(model, test_loader)

    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")

if __name__ == '__main__':
    main()
