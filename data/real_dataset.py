import torch
import torchvision

from utils.parameters import args_parser
args = args_parser()

def download_dataset(dataset_name = None):
    if dataset_name == 'MNIST':
        # 这个函数包括了两个操作：将图片转换为张量，以及将图片进行归一化处理
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])

        # 加载训练和测试数据
        train_dataset = torchvision.datasets.MNIST('./data/dataset/MyMNIST/', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.MNIST('./data/dataset/MyMNIST/', train=False, transform=transform, download=True)

        # 建立一个数据迭代器
        # 装载训练集
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.batchsize,
                                                   shuffle=True)
        # 装载测试集
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=args.batchsize,
                                                  shuffle=True)
        return train_loader, test_loader


