import importlib
import yaml

import torch
import torchvision

def get_object(name, kwargs):
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    obj_class = getattr(module, class_name, None)
    if obj_class is not None:
        return obj_class(**kwargs)
    else:
        return None

class Config():
    
    def __init__(self, config_path) -> None:
        with open(config_path) as f:
            self.configs = yaml.load(f, Loader=yaml.Loader)

    def get_model(self):
        model = get_object(
            self.configs["model"]["name"], 
            self.configs["model"]["kwargs"]
        )
        return model
    
    def get_loader(self):
        dataloaders = {}
        for phase, phase_config in self.configs["data"].items():
            subsets = []
            for dataset_config in phase_config["datasets"]:
                trans_list = []
                trans_base = importlib.import_module(dataset_config["transforms"]["base"])
                for trans_conf in dataset_config["transforms"]["operations"]:
                    trans_list.append(getattr(trans_base, trans_conf["name"])(**trans_conf["kwargs"]))
                dataset_kwargs = dataset_config["kwargs"]
                dataset_kwargs["transform"] = getattr(trans_base, "Compose")(trans_list)
                subsets.append(get_object(dataset_config["name"], dataset_kwargs))
            dataset = torch.utils.data.ConcatDataset(subsets)
            loader_kwargs = phase_config["dataloader"]["kwargs"]
            loader_kwargs["dataset"] = dataset
            dataloaders[phase+"_loader"] = get_object(phase_config["dataloader"]["name"], loader_kwargs)
        return dataloaders
    
    def get_optimizer(self, params):
        kwargs = self.configs["train"]["optimizer"]["kwargs"]
        kwargs["params"] = params
        return get_object(self.configs["train"]["optimizer"]["name"], kwargs)
    
    def get_scheduler(self, optimizer):
        kwargs = self.configs["train"]["scheduler"]["kwargs"]
        kwargs["optimizer"] = optimizer
        return get_object(self.configs["train"]["scheduler"]["name"], kwargs)
    
    def get_criterion(self):
        return get_object(self.configs["train"]["criterion"]["name"], self.configs["train"]["criterion"]["kwargs"])

    def use_cuda(self):
        return self.configs["use_cuda"]

    def epochs(self):
        return self.configs["train"]["epochs"]

    def get_pretrain_configs(self):
        return {
            "path": self.configs["pretrain"]["model_path"],
            "model_only": self.configs["pretrain"]["model_only"]
        }