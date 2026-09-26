import os
import json
import time

from torch_geometric.utils import degree
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import  DataLoader
from tqdm import tqdm

from utils.utils import save_checkpoint, emergency_checkpoint, \
    load_emergency_checkpoint_multi
from loss_functions.loss_functions import train_loss,evaluation_score,loss_wrapper
from optimizers.optimizers import train_optimizers,train_schedulers
from models.models import Resilience_model

def run_training( trainset, validationset,train_param,net_param,loss_param,optimizer_param,records_path):
    # 评价指标 记录变量初始化

    # 加载数据集
   # trainset_data = torch.cat(trainset, dim=0)  # n*pred_len,F
  #  validationset_data = torch.cat(validationset["data"], dim=0)  # n*pred_len,F

    train_loader = DataLoader(trainset, batch_size=train_param["train_batch_size"], shuffle=True, drop_last=False)
    val_loader   = DataLoader(validationset, batch_size=train_param["val_batch_size"], shuffle=False, drop_last=False)

    #设置训练模型参数、优化器参数、损失函数参数
    net_param['log_var']=loss_param["log_var"]
    net_param["degree_loss_weight"]=loss_param["degree_loss_weight"]
    train_GNN = Resilience_model(net_param=net_param
                                 ).to(net_param["device"])
    if loss_param["loss_type"] == "prediction":
        task_cond = torch.tensor([0]).to(train_GNN.device)
    elif loss_param["loss_type"] == "reconstruction":
        task_cond = torch.tensor([1]).to(train_GNN.device)
    elif loss_param["loss_type"] == "multitask":
        task_cond = torch.tensor([2]).to(train_GNN.device)
    else:
        raise "error: model task don't exist"

    optimizer = train_optimizers(filter(lambda p: p.requires_grad,train_GNN.parameters()),optimizer_param)
    if optimizer_param["scheduler_set"]==True:
        scheduler= train_schedulers(optimizer,optimizer_param)
    else:
        scheduler=None
    score_metrics = ["predictability", "reconstructability", "multitask"]
    #设置模型性能评价指标参数
    current_step,record_scores = load_emergency_checkpoint_multi(checkpoint_path=records_path,
                                             model=train_GNN,
                                             optimizer=optimizer,
                                             device=net_param["device"],
                                             scheduler=scheduler,
                                            score_metrics=score_metrics
    )
    init_epoch = current_step

    accumulation_steps = 1
    #开始训练
    try:
        for epoch in range(init_epoch,train_param["train_epochs"]):
            train_GNN.train()
            # 模型性能评价指标值 初始化
            train_score_pred=0
            train_score_recon=0
            train_score_multi=0
            val_score_pred = 0
            val_score_recon = 0
            val_score_multi = 0

            #批量训练
            for n,data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):

               # print("\r******************train_epoch:{} batch:{}********************\n".format(epoch,n),end=" ")


                data = data.to(net_param["device"])
                loss_list,task_train = train_GNN.training_step(gdata=data,task_cond=task_cond)
                loss=loss_list[task_train]
                if  torch.isnan(loss).any():
                    print("loss is None")
                    continue

                # time_train=time.time()
                # print("train time_cost:{}".format(time_train-time_step))

                loss.backward()

                torch.cuda.empty_cache()
                if (n + 1) % accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(train_GNN.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()  # Clear gradients.
                train_score_pred=n*train_score_pred/(n+1)+loss_list[0].cpu().detach().item()/(n+1)
                train_score_recon = n * train_score_recon / (n + 1) + loss_list[1].cpu().detach().item() / (n + 1)
                train_score_multi = n * train_score_multi / (n + 1) + loss_list[2].cpu().detach().item() / (n + 1)

            if  torch.isnan(loss).any():
                raise ValueError("loss is None")
            if optimizer_param["scheduler_set"] == True:
                scheduler.step()
            current_step = epoch + 1
            # print("\rtrain_epoch:{}".format(epoch),end=" ")
            # print("train_loss:{}".format(train_score))
            #validation
            with torch.no_grad():
                if  train_param["test_set"]:

                    train_GNN.eval()
                    for n, data in enumerate(val_loader):
                        data = data.to(net_param["device"])
                        loss_list,task_train = train_GNN.training_step(gdata=data,task_cond=task_cond)


                        val_score_pred = n * val_score_pred / (n + 1) + loss_list[0].cpu().detach().item() / (n + 1)
                        val_score_recon = n * val_score_recon / (n + 1) + loss_list[1].cpu().detach().item() / (n + 1)
                        val_score_multi = n * val_score_multi / (n + 1) + loss_list[2].cpu().detach().item() / (n + 1)


            print("val_score_multi_loss:{}".format(val_score_multi))
            record_scores["epoch"].append(epoch)

            record_scores["predictability"]["train_scores"].append(train_score_pred)
            record_scores["predictability"]["val_scores"].append(val_score_pred)
            record_scores["reconstructability"]["train_scores"].append(train_score_recon)
            record_scores["reconstructability"]["val_scores"].append(val_score_recon)
            record_scores["multitask"]["train_scores"].append(train_score_multi)
            record_scores["multitask"]["val_scores"].append(val_score_multi)

            #save model parameters in training process
            if epoch%train_param["ckpt_period"]==0 and epoch!=0 and train_param["ckpt"]:
                sckpt_path =os.path.join(records_path,"ckpt")
                if os.path.exists(sckpt_path):
                    print("ckpt文件夹目录已存在")
                    pass
                else:
                    os.mkdir(sckpt_path)

                save_checkpoint(path=sckpt_path, model_name="tmpt_model_{}iter".format(epoch), model=train_GNN,
                                net_param=net_param)

    except Exception as e:
     #   print(e)
        if "CUDA out of memory" in str(e):
            # 首先，清理当前可能持有引用的计算图
            loss = None
            # 强制Python的垃圾回收
            import gc
            gc.collect()
            # 清空PyTorch的CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        emergency_checkpoint(train_GNN, net_param, optimizer, scheduler, current_step,record_scores,
                             checkpoint_path=records_path)
        data_path = os.path.join(records_path, "train_trace")
        if os.path.exists(data_path):
            print("warning:train_trace文件夹目录已存在")
        else:
            os.mkdir(data_path)
        with open(data_path + '/record_scores.json', 'w') as f:
            json.dump(record_scores, f, indent=4, separators=(',', ':'))
        print(e)
        raise
    #save fininal trainde model 文件夹可略去
    model_path = os.path.join(records_path,"trained_model")
    if os.path.exists(model_path):
        print("trained_model文件夹目录已存在")
    else:
        os.mkdir(model_path)

    save_checkpoint(path=model_path, model_name="model_trained", model=train_GNN,
                    net_param=net_param)

    #保存训练模型性能指标数据 文件夹可略去
    data_path = os.path.join(records_path, "train_trace")
    if os.path.exists(data_path):
        print("warning:train_trace文件夹目录已存在")
    else:
        os.mkdir(data_path)
    with open(data_path+'/record_scores.json', 'w') as f:
        json.dump(record_scores, f, indent=4, separators=(',', ':'))
    return record_scores









