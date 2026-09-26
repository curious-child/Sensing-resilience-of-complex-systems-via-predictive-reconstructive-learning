import json
import multiprocessing

import torch_geometric
import yaml
from scipy.fftpack import ifftn

from configs.configs_diffusion_multi import parse_args
import os
import numpy as np

from utils.utils import save_config, grid_parameters_generative_learning, pre_DataSet_spdata, \
    grid_parameters_generative_learning_VAE
from sklearn.model_selection import train_test_split,KFold


import itertools as it
#from train.train_vanilla import run_training
from train.train_multisk_recon_repred import run_training
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from utils.data_visualization import  model_evaluation_metrics_curves_vanilla
import random
import torch

def seed_torch(seed=1029):
    random.seed(seed)   # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)   # numpy的随机性
    torch.manual_seed(seed)   # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)   # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False   # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True   # 选择确定性算法


# 留出法 模型性能评估
def hold_out_score(dataset,train_param,net_param,loss_param,optimizer_param,records_path,configs_counts=0,vision_show=True):
    save_data_path = os.path.join(records_path,"hold_out")
    if os.path.exists(save_data_path):
        print("waring:hold out文件夹目录已存在")
    else:
        os.mkdir(save_data_path)

    trainset,validationset = train_test_split(dataset,train_size=train_param["traindata_size"])

    record_scores=run_training(trainset=trainset,

                             validationset=validationset,
                             train_param=train_param,
                             net_param=net_param,
                             loss_param=loss_param,
                             optimizer_param=optimizer_param,
                             records_path=save_data_path)
    # record_scores_path=os.path.join(save_data_path,"train_trace")
    # with open(record_scores_path + '/record_scores.json', 'r') as f:
    #     record_scores=json.load(f)
    #可视化 基于不同模型性能指标的训练轨迹 此处可拓展 也可略去
    if vision_show==True:
       score_metrics=["predictability","reconstructability","multitask"]
       fig = plt.figure("metric—curves_configs_{}".format(configs_counts))
       model_evaluation_metrics_curves_vanilla(fig=fig,

                                       record_scores=record_scores,
                                               score_metrics=score_metrics)

       fig.savefig(save_data_path+"/metric—curves_configs_{}".format(configs_counts))
       plt.close(fig)
    return record_scores

# 交叉验证 模型性能评估
#未修改 暂略


def grid_search(dataset_params,train_params,net_params,loss_params,optimizer_params,records_path):
    grid_search_path = records_path + "/grid_search"
    if os.path.exists(grid_search_path):
        print("waring:grid_search文件夹目录已存在")
    else:
        os.mkdir(grid_search_path)
    configs_count=0
    configs_record_scores=list()

    for params_values in it.product(  *dataset_params.values() ):
        dataset_param=dict(zip(dataset_params.keys(),params_values))
        dataset,max_degree_dataset = pre_DataSet_spdata(**dataset_param)

        parameters_list=grid_parameters_generative_learning_VAE(train_params,net_params,loss_params,optimizer_params)

        for train_param,net_param,loss_param,optimizer_param in parameters_list:

            seed_torch(123)
            save_config_path = os.path.join(grid_search_path , "config_{}".format(configs_count))
            if os.path.exists(save_config_path):
                print("waring:config文件夹目录已存在")
            else:
                os.mkdir(save_config_path)
            # 保存每一个训练模型参数,并检查是否已训练
            Not_train_flag=save_config(path=save_config_path, configs_name="config_{}.yaml".format(configs_count),
                        dataset_param=dataset_param, train_param=train_param,
                        net_param=net_param, loss_param=loss_param, optimizer_param=optimizer_param)
            if Not_train_flag:
                if train_param["model_evaluation"]=="hold_out":
                    net_param["max_degree_dataset"]=max_degree_dataset
                    net_param["Dy_transformer"]["windows"]=dataset_param["windows"]
                    net_param["Dy_transformer"]["dataset_nf"] = 1

                    record_scores= hold_out_score(dataset=dataset,train_param= train_param, net_param=net_param,loss_param= loss_param, optimizer_param=optimizer_param,
                                   records_path=save_config_path,configs_counts=configs_count)

                else:
                    raise ValueError("the definition of model_evaluation don't exit\n"
                                     "\tyou can define it before using it")
                configs_record_scores.append(record_scores)
            #
            configs_count+=1
            #各模型性能评价指标记录，用于分析模型超参数选择




    #

    with open(grid_search_path + '/configs_record_scores.json', 'w') as f:
        json.dump(configs_record_scores, f, indent=4, separators=(',', ':'))





if __name__ == '__main__':
    args = parse_args()

    with open(args.cfg,'r') as f:
        config_params=yaml.safe_load(f)
    dataset_param=config_params["dataset"]
    train_param =config_params["train"]
    net_param   =config_params['net']
    loss_param  =config_params['loss']
    optimizer_param=config_params['optimizer']
   # fig = plt.figure("metric—curves_configs_{}".format(1))
    records_path=config_params["out_dir"]
    if os.path.exists(records_path):
        print("records_path:{}文件夹目录已存在".format(records_path))
    else:
        os.mkdir(records_path)

    if args.train_mode == "grid":
        grid_search(dataset_params=dataset_param,train_params=train_param,
                net_params=net_param, loss_params=loss_param,
                optimizer_params=optimizer_param, records_path=records_path)
        # parallel_grid_search(dataset_params=dataset_param, train_params=train_param,
        #             net_params=net_param, loss_params=loss_param,
        #             optimizer_params=optimizer_param, records_path=records_path)
    else:
        raise ValueError("the definition of train_mode don't exit\n")