import argparse
import os
import tqdm
import torch
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from dlmodels.utils.config import Config

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="configs/vggnet/vggnet_train.yaml")
    parser.add_argument("--save_path", default="/home/sensetime/DL/train_folder/vgg")
    return parser.parse_args()

def train(config):
    model = config.get_model()
    print(model)

    if config.use_cuda():
        model = model.cuda()

    dataloaders = config.get_loader()
    optimizer = config.get_optimizer(model.parameters())
    scheduler = config.get_scheduler(optimizer)
    criterion = config.get_criterion()

    step = 0
    best_acc = 0

    writer = SummaryWriter(os.path.join(args.save_path, "log"))

    for i in range(config.epochs()):
        model.train()
        epoch_loss = 0
        epoch_correct = 0

        for input, label in tqdm.tqdm(dataloaders["train_loader"]):
            if config.use_cuda():
                input = input.cuda()
                label = label.cuda()

            output = model(input)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * input.size(0)
            _, preds = torch.max(output, 1)
            epoch_correct += torch.sum(preds == label.data)
            writer.add_scalar("train/loss_step", loss.item(), step)
            step += 1
        
        epoch_loss = epoch_loss / len(dataloaders["train_loader"].dataset)
        epoch_acc = epoch_correct / len(dataloaders["train_loader"].dataset)
        writer.add_scalar("train/loss_epoch", epoch_loss, i)
        writer.add_scalar("train/acc_epoch", epoch_acc.item(), i)

        model.eval()
        epoch_loss = 0
        epoch_correct = 0

        for input, label in tqdm.tqdm(dataloaders["val_loader"]):
            if args.use_cuda:
                input = input.cuda()
                label = label.cuda()
            with torch.no_grad():
                output = model(input)
                loss = criterion(output, label)
                _, preds = torch.max(output, 1)

            epoch_loss += loss.item() * input.size(0)
            epoch_correct += torch.sum(preds == label.data)
        
        epoch_loss = epoch_loss / len(dataloaders["val_loader"].dataset)
        epoch_acc = epoch_correct / len(dataloaders["val_loader"].dataset)

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), os.path.join(args.save_path, "best.pt"))

        writer.add_scalar("valid/loss_epoch", epoch_loss, i)
        writer.add_scalar("valid/acc_epoch", epoch_acc.item(), i)

        for id, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar("lr/param_group%d"%id, param_group["lr"], i)
        pass

if __name__ ==  "__main__":
    args = arg_parser()
    
    config = Config(args.config_path)
    train(config)
    