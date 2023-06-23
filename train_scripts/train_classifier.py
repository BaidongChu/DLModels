import argparse
import os
import tqdm
import torch
import time
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from dlmodels.utils.config import Config

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="configs/vggnet/resnet_train.yaml")
    parser.add_argument("--save_path", default="/home/sensetime/DL/train_folder/resnet")
    return parser.parse_args()

def save(save_path, epoch, model, best_metric, optimizer, scheduler):
    save_dict = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_metric": best_metric
    }
    torch.save(save_dict, save_path)

def load(load_path, model, optimizer=None, scheduler=None, ):
    ckpt = torch.load(load_path)
    model.load_state_dict(ckpt["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["best_metric"]

def train(config):
    model = config.get_model()
    print(model)
    if config.use_cuda():
        model = model.cuda()

    dataloaders = config.get_loader()
    optimizer = config.get_optimizer(model.parameters())
    scheduler = config.get_scheduler(optimizer)
    criterion = config.get_criterion()

    epoch = 0
    step = 0
    best_acc = 0

    pretrain_config = config.get_pretrain_configs()
    if pretrain_config["path"] != "":
        if pretrain_config["model_only"]:
            epoch, best_acc = load(pretrain_config["path"], model)
        else:
            epoch, best_acc = load(pretrain_config["path"], model, optimizer, scheduler)
    
    date = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_path = os.path.join(args.save_path, date)
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(os.path.join(save_path, "log"))

    while epoch < config.epochs():
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
        
        for id, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar("lr/param_group%d"%id, param_group["lr"], epoch)

        scheduler.step()
        epoch_loss = epoch_loss / len(dataloaders["train_loader"].dataset)
        epoch_acc = epoch_correct / len(dataloaders["train_loader"].dataset)
        writer.add_scalar("train/loss_epoch", epoch_loss, epoch)
        writer.add_scalar("train/acc_epoch", epoch_acc.item(), epoch)

        model.eval()
        epoch_loss = 0
        epoch_correct = 0

        for input, label in tqdm.tqdm(dataloaders["val_loader"]):
            if config.use_cuda():
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
            save(os.path.join(args.save_path, "best.ckpt"), epoch, model, best_acc, optimizer, scheduler)

        writer.add_scalar("valid/loss_epoch", epoch_loss, epoch)
        writer.add_scalar("valid/acc_epoch", epoch_acc.item(), epoch)
        
        if epoch % 1 == 0:
            save(os.path.join(args.save_path, "epoch_%d.ckpt"%epoch), epoch, model, best_acc, optimizer, scheduler)
        

if __name__ ==  "__main__":
    args = arg_parser()
    
    config = Config(args.config_path)
    train(config)
    