from utils import args,get_test_dataloader
import torch
import os
import torch.backends.cudnn as cudnn
from resnet_18 import ResNet18
# 测试函数

def testing(model):
    model.eval()
    correct = 0.0
    total = 0.0
    test_loader = get_test_dataloader()

    for images, labels in test_loader:
        if args['cuda']:
            images = images.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            pred = model(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    test_acc = correct / total
    model.train()
    return test_acc



def testing15(model, topk=(1,)):
    model.eval()
    correct = [0.0] * len(topk)
    total = 0.0
    test_loader = get_test_dataloader()

    for images, labels in test_loader:
        if args['cuda']:
            images = images.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            pred = model(images)

        _, pred_topk = pred.topk(max(topk), 1, True, True)
        pred_topk = pred_topk.t()
        correct_batch = pred_topk.eq(labels.view(1, -1).expand_as(pred_topk))

        total += labels.size(0)

        for i, k in enumerate(topk):
            correct[i] += correct_batch[:k].reshape(-1).float().sum(0)  # 使用 reshape 方法

    test_acc = [c / total for c in correct]
    model.train()
    return test_acc




def get_model_acc(load_dir, topk=(1,)):
    model = ResNet18(num_classes=100)
    if args['cuda']:
        model = model.cuda()
    model.load_state_dict(torch.load(load_dir))
    
    test_acc = testing15(model, topk=topk)
    return test_acc



if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    final_dir = os.path.join(results_dir, 'final')

    model1_dir = os.path.join(final_dir, 'zero_CIFAR100_ResNet18.pth')

    args['cuda'] = torch.cuda.is_available()
    cudnn.benchmark = True

    torch.manual_seed(0)
    if args['cuda']:
        torch.cuda.manual_seed(0)
        
    top1_acc, top5_acc = get_model_acc(model1_dir, topk=(1, 5))
    print("Top-1 Accuracy:", top1_acc.item())
    print("Top-5 Accuracy:", top5_acc.item())


