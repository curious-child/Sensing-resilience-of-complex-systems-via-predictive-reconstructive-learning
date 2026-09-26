import numpy as np
import torch
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional, Callable
def train_optimizers(GNN_parameters,optimizer_param):
    #优化器种类不作为 网格搜索参数
    if optimizer_param["optimizer_name"]== "Adam":
        return torch.optim.Adam(GNN_parameters,lr=optimizer_param["lr"],weight_decay=float(optimizer_param["weight_decay"]))

    elif optimizer_param["optimizer_name"] == "SGD":
        return torch.optim.SGD(GNN_parameters,lr=optimizer_param["lr"],momentum=float(optimizer_param["momentum"]),weight_decay=float(optimizer_param["weight_decay"]))

    else:
        raise ValueError("the definition of optimier don't exit\n"
                         "\tyou can define it before using it")

def train_schedulers(optimizer,optimizer_param):
    # 学习速率 https://zhuanlan.zhihu.com/p/494010639
    # 学习速率调节器 种类不作为 网格搜索参数
    if optimizer_param["scheduler"]=="StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=optimizer_param["stepLR_stepsize"],gamma=optimizer_param["stepLR_gamma"])
    elif optimizer_param["scheduler"]=="MultiStepLR":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=optimizer_param["MstepLR_milestones"], gamma=optimizer_param["MstepLR_gamma"])
    elif optimizer_param["scheduler"] == "CyclicLR":
        return torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=optimizer_param["CyclicLR_blr"], max_lr=optimizer_param["CyclicLR_mlr"], step_size_up=optimizer_param["CyclicLR_upsteps"],mode=optimizer_param["CyclicLR_mode"], gamma=optimizer_param["CyclicLR_mode"])
    elif optimizer_param["scheduler"] == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=optimizer_param["CALR_Tmax"],eta_min=optimizer_param["CALR_minlr"])
    else:
        raise ValueError("the definition of scheduler don't exit\n"
                         "\tyou can define it before using it")



