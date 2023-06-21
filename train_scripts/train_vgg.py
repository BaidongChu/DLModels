import argparse
import yaml
import copy
import os
import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

from dlmodels.models.classifiers.vgg_net import VGGNet

def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", default="configs/vggnet/vggnet_16.yaml")
    parser.add_argument("--save_path", default="/home/sensetime/DL/train_folder/vgg")
    parser.add_argument("--use_cuda", action="store_true", default=True)

    return parser.parse_args()


def get_cifar_loader(configs):
    batch_size = configs["dataloader"]["batchsize"]

    train_trans_conf = configs["dataloader"]["transforms"]["train"]
    train_trans = []
    for trans_info in train_trans_conf:
        trans = getattr(transforms, trans_info["type"])
        kwargs = copy.deepcopy(trans_info)
        kwargs.pop("type")
        train_trans.append(trans(**kwargs))
    train_trans = transforms.Compose(train_trans)
    
    valid_trans_conf = configs["dataloader"]["transforms"]["val"]
    valid_trans = []
    for trans_info in valid_trans_conf:
        trans = getattr(transforms, trans_info["type"])
        kwargs = copy.deepcopy(trans_info)
        kwargs.pop("type")
        valid_trans.append(trans(**kwargs))
    valid_trans = transforms.Compose(valid_trans)

    train_set = datasets.CIFAR100('/home/sensetime/DL/data/cifar-100', True, train_trans)
    valid_set = datasets.CIFAR100('/home/sensetime/DL/data/cifar-100', False, valid_trans)

    train_loader = DataLoader(train_set, batch_size, True)
    valid_loader = DataLoader(valid_set, batch_size, False)

    # for img, label in train_loader:
    #     grid_img = torchvision.utils.make_grid(img, 4)
    #     # plt.imshow(grid_img.permute(1, 2, 0))
    #     # plt.show()
    #     continue
    return train_loader, valid_loader


if __name__ ==  "__main__":
    args = arg_parser()
    
    with open(args.config_file) as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    model = VGGNet(configs)
    #model = torchvision.models.vgg19()
    print(model)

    if args.use_cuda:
        model = model.cuda()

    train_loader, valid_loader = get_cifar_loader(configs)
    step = 0

    best_acc = 0

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.5)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(os.path.join(args.save_path, "log"))

    for i in range(configs["train"]["epochs"]):
        model.train()
        epoch_loss = 0
        epoch_correct = 0

        for input, label in tqdm.tqdm(train_loader):
            
            if args.use_cuda:
                input = input.cuda()
                label = label.cuda()

            output = model(input)
            loss = criterion(output, label)

            module = model.classifier[6]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(output, 1)
            epoch_loss += loss.item() * input.size(0)
            epoch_correct += torch.sum(preds == label.data)

            writer.add_scalar("train/loss_step", loss.item(), step)
            step += 1
        
        epoch_loss = epoch_loss / len(train_loader.dataset)
        epoch_acc = epoch_correct / len(train_loader.dataset)
        writer.add_scalar("train/loss_epoch", epoch_loss, i)
        writer.add_scalar("train/acc_epoch", epoch_acc, i)

        model.eval()
        epoch_loss = 0
        epoch_correct = 0

        for input, label in valid_loader:
            if args.use_cuda:
                input = input.cuda()
                label = label.cuda()
            with torch.no_grad():
                output = model(input)
                loss = criterion(output, label)
                _, preds = torch.max(output, 1)

            epoch_loss += loss.item() * input.size(0)
            epoch_correct += torch.sum(preds == label.data)
        
        epoch_loss = epoch_loss / len(valid_loader.dataset)
        epoch_acc = epoch_correct / len(valid_loader.dataset)

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), os.path.join(args.save_path, "best.pt"))

        writer.add_scalar("valid/loss_epoch", loss.item(), i)
        writer.add_scalar("valid/acc_epoch", epoch_acc.item(), i)
        pass