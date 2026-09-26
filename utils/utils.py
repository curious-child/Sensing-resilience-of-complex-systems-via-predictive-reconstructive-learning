import copy
import json
import math
import os
import random

from easydict import EasyDict
from scipy import stats
import itertools as it
import igraph
import numpy as np
import networkx as nx
import torch_geometric
import torch_geometric.utils as tg_utils
from torch_geometric.data import Data, Batch, HeteroData
import torch
import numpy as np
from igraph import *
import matplotlib.pyplot as plt
from glob import glob
from sklearn.manifold import TSNE
import yaml
from torch_geometric.utils import k_hop_subgraph

from models.models import Resilience_model


def grid_parameters_supervised_learning(train_params,net_params,loss_params,optimizer_params,**kwargs):
    parameters_list=[]
    for params_values in it.product(*loss_params.values()):
        loss_param = dict(zip(loss_params.keys(), params_values))
        for params_values in it.product(*train_params.values()):
            train_param = dict(zip(train_params.keys(), params_values))
            #  print(train_param)

            net_params_copy = net_params.copy()
            net_params_copy.pop("prelayers_gnn_param")
            net_params_copy.pop("enclayers_gnn_param")
            #net_params_copy.pop("mask_model")
            for params_values in it.product(*net_params_copy.values()):
                net_param = dict(zip(net_params_copy.keys(), params_values))

                # print(net_param["prelayers_gnn"])
                # print(net_param["enclayers_gnn"])
                for prelayers_values in it.product(*net_params["prelayers_gnn_param"][net_param["prelayers_gnn"]].values()):
                    net_param["prelayers_gnn_param"] = dict(
                        zip(net_params["prelayers_gnn_param"][net_param["prelayers_gnn"]].keys(), prelayers_values))
                    # print(net_param["prelayers_gnn_param"])
                    for enclayers_values in it.product(
                            *net_params["enclayers_gnn_param"][net_param["enclayers_gnn"]].values()):
                        net_param["enclayers_gnn_param"] = dict(
                            zip(net_params["enclayers_gnn_param"][net_param["enclayers_gnn"]].keys(), enclayers_values))

                        for params_values in it.product(*optimizer_params.values()):
                            optimizer_param = dict(zip(optimizer_params.keys(), params_values))
                            parameters_list.append((train_param,net_param,loss_param,optimizer_param))

    return parameters_list
def grid_parameters_generative_learning_VAE(train_params, net_params, loss_params, optimizer_params, **kwargs):
    parameters_list = []

    for params_values in it.product(*train_params.values()):
        train_param = dict(zip(train_params.keys(), params_values))
        #  print(train_param)

        net_params_copy = net_params.copy()
        net_params_copy.pop("Dy_transformer")

        net_params_copy.pop("Hyp_recon_model") if "Hyp_recon_model" in net_params.keys() else None
        for params_values in it.product(*net_params_copy.values()):
            net_param = dict(zip(net_params_copy.keys(), params_values))
            for Dy_transformer_values in it.product(*net_params["Dy_transformer"].values()):
                net_param["Dy_transformer"] = dict(
                    zip(net_params["Dy_transformer"].keys(), Dy_transformer_values))

                if "Hyp_recon_model" in net_params.keys():
                    for Hyp_recon_model_values in it.product(
                            *net_params["Hyp_recon_model"].values()):
                        net_param["Hyp_recon_model"]=dict(zip(net_params["Hyp_recon_model"].keys(), Hyp_recon_model_values))
                        for params_values in it.product(*loss_params.values()):
                            loss_param = dict(zip(loss_params.keys(), params_values))
                            #  print("policy gradient:", loss_param["sample"])
                            for params_values in it.product(*optimizer_params.values()):
                                optimizer_param = dict(zip(optimizer_params.keys(), params_values))
                                parameters_list.append((copy.deepcopy(train_param), copy.deepcopy(net_param), copy.deepcopy(loss_param), copy.deepcopy(optimizer_param)))
                else:
                    for params_values in it.product(*loss_params.values()):
                        loss_param = dict(zip(loss_params.keys(), params_values))
                        #  print("policy gradient:", loss_param["sample"])
                        for params_values in it.product(*optimizer_params.values()):
                            optimizer_param = dict(zip(optimizer_params.keys(), params_values))
                            parameters_list.append((copy.deepcopy(train_param), copy.deepcopy(net_param),
                                                    copy.deepcopy(loss_param), copy.deepcopy(optimizer_param)))


    return parameters_list
def record_scores_analysis(configs_record_scores,metrics,train_validation,score_thershold):
    best_scores = float('-inf')
    best_config = None
    score_thershold_config = dict()
    for config_name,record_scores in configs_record_scores.items():
        cmp_best_score = max(record_scores[metrics][train_validation])
        if cmp_best_score > best_scores:
            best_scores = cmp_best_score
            best_config = config_name
        if cmp_best_score >= score_thershold:
            score_thershold_config[config_name] = cmp_best_score
    print("Best config:", best_config)
    print("Best scores:", best_scores)
    for n,config_name in enumerate(score_thershold_config.keys()):

        print("Score thershold config {}:\n".format(n), config_name)
    return best_config, best_scores,score_thershold_config
def grid_parameters_generative_learning(train_params, net_params, loss_params, optimizer_params, **kwargs):
    parameters_list = []

    for params_values in it.product(*train_params.values()):
        train_param = dict(zip(train_params.keys(), params_values))
        #  print(train_param)

        net_params_copy = net_params.copy()
        net_params_copy.pop("Dy_transformer")
        net_params_copy.pop("Gg_diffusion_model") if "Hyp_recon_model" in net_params.keys() else None
        net_params_copy.pop("Hyp_recon_model") if "Hyp_recon_model" in net_params.keys() else None
        for params_values in it.product(*net_params_copy.values()):
            net_param = dict(zip(net_params_copy.keys(), params_values))
            for Dy_transformer_values in it.product(*net_params["Dy_transformer"].values()):
                net_param["Dy_transformer"] = dict(
                    zip(net_params["Dy_transformer"].keys(), Dy_transformer_values))

                for Gg_diffusion_model_values in it.product(
                        *net_params["Gg_diffusion_model"].values()):
                    net_param["Gg_diffusion_model"] = dict(
                        zip(net_params["Gg_diffusion_model"].keys(), Gg_diffusion_model_values))
                    if "Hyp_recon_model" in net_params.keys():
                        for Hyp_recon_model_values in it.product(
                                *net_params["Hyp_recon_model"].values()):
                            net_param["Hyp_recon_model"]=dict(zip(net_params["Hyp_recon_model"].keys(), Hyp_recon_model_values))
                            for params_values in it.product(*loss_params.values()):
                                loss_param = dict(zip(loss_params.keys(), params_values))
                                #  print("policy gradient:", loss_param["sample"])
                                for params_values in it.product(*optimizer_params.values()):
                                    optimizer_param = dict(zip(optimizer_params.keys(), params_values))
                                    parameters_list.append((copy.deepcopy(train_param), copy.deepcopy(net_param), copy.deepcopy(loss_param), copy.deepcopy(optimizer_param)))
                    else:
                        for params_values in it.product(*loss_params.values()):
                            loss_param = dict(zip(loss_params.keys(), params_values))
                            #  print("policy gradient:", loss_param["sample"])
                            for params_values in it.product(*optimizer_params.values()):
                                optimizer_param = dict(zip(optimizer_params.keys(), params_values))
                                parameters_list.append((copy.deepcopy(train_param), copy.deepcopy(net_param),
                                                        copy.deepcopy(loss_param), copy.deepcopy(optimizer_param)))


    return parameters_list
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
def gen_graph(g_type, num_min=20, num_max=40):
    max_n = num_max
    min_n = num_min
    cur_n = np.random.randint(max_n - min_n + 1) + min_n

    if g_type == 'erdos_renyi':
        while True:
            ER_p=random.uniform(0.1,0.9)
            g = Graph.Erdos_Renyi(n=cur_n,p=ER_p,directed=False)
            if g.is_connected():
                break
    elif g_type == 'small-world':
        while True:
            nei_max=round(0.35*cur_n)
            nei_min=round(0.15*cur_n)
            WS_nei=random.randint(nei_min,nei_max)
            WS_p = random.uniform(0, 0.15)
            g = Graph.Watts_Strogatz(dim=1,size=cur_n,nei=WS_nei,p=WS_p)#可包含最近邻规则图（p=0）
            if g.is_connected():
                break
    elif g_type == 'barabasi_albert':
        while True:
            m_max = round(0.25 * cur_n)
            m_min = round(0.1 * cur_n)
            BA_m = random.randint(m_min, m_max)
            g = Graph.Barabasi(n=cur_n, m=BA_m,directed=False)
            if g.is_connected():

                break
    elif g_type=="static_power_law":
        while True:
            exp=random.uniform(2,3)
            pl_max = round(0.25*cur_n* cur_n)
            pl_min = round(0.05*cur_n* cur_n)
            pl_edges=random.randint(pl_min,pl_max)
            g = Graph.Static_Power_Law(n=cur_n,m=pl_edges,exponent_out=exp)
            if g.is_connected():
                break
    elif g_type == "K_Regular":
        while True:
            k_regular=random.randint(round(0.2*cur_n),cur_n-2)  #可包含完全图（cur-1）
            if k_regular*cur_n %2==0 and cur_n>=(k_regular+1) : #k*n必须为偶数 且k+1 ≤n
                g = Graph.K_Regular(n=cur_n, k=k_regular)
                if g.is_connected():
                    break

    return g

def to_gnngraph(g,features,targets=None,SL=True):

    if "None" in features:
        x = np.ones((g.vcount(), 1))
    elif "one-hot" in features:
        x=np.identity(g.vcount())
    else:
        x = np.column_stack(
            tuple(
                g.vs[feature] for feature in features
            )
        )
    x = torch.from_numpy(x).to(torch.float)
    if SL:
        y = g.vs[targets]
        y = torch.tensor(y).to(torch.float)
    else:
        # G=igraph.Graph.to_networkx(g)
        # y=[pre_graph_op(G)]
        y=None


    source_nodes1 = [edge.source for edge in g.es]
    target_nodes1 = [edge.target for edge in g.es]
    source_nodes = source_nodes1 + target_nodes1
    target_nodes = target_nodes1 + source_nodes1

    return Data(x=x,y=y,edge_index=torch.tensor([source_nodes,target_nodes],dtype=torch.long))

def graph_properties(gnn_g,targets):
    node_v = 1 - gnn_g.x
    node_mask = torch.bernoulli(node_v).to(torch.bool)
    nodes_to_keep = torch.nonzero(node_mask).squeeze(1)
    subgraph = gnn_g.subgraph(nodes_to_keep)
    subgraph =tg_utils.to_networkx(subgraph, to_undirected=True)
    net_G = tg_utils.to_networkx(gnn_g, to_undirected=True)
    if targets == "LCC":
        return len(max(nx.connected_components(subgraph), key=len))
    elif targets == "global_CC":
        return nx.transitivity(subgraph)
    elif targets == "average_CC":
        return nx.average_clustering(subgraph)
    elif targets == "natural_connectivity":
        estrada_index = nx.estrada_index(subgraph)
        # 自然连通性
        n = nx.number_of_nodes(subgraph)
        return np.log(estrada_index / n)
    elif targets=="global_efficiency":
        return nx.global_efficiency(subgraph)
    elif targets=="density":
        return nx.density(subgraph)
    else:
        raise ValueError("Error fitness_func_type:{}".format(targets))
def to_gnngraph_nx(g,features,targets=None,SL=True):

    if "None" in features:
        gnn_g=torch_geometric.utils.from_networkx(g,group_node_attrs=None)
        gnn_g.x = torch.ones(gnn_g.num_nodes, 1).float()

    elif "random" in features:
        gnn_g = torch_geometric.utils.from_networkx(g, group_node_attrs=None)
        x0 = torch.rand(gnn_g.num_nodes)
        gnn_g.x = x0.reshape(-1,).float()
    else:
        gnn_g = torch_geometric.utils.from_networkx(g, group_node_attrs=features)
    if SL:

        if targets in g.nodes[0].keys():
            y_dict = nx.get_node_attributes(g,targets)
            gnn_g.y = torch.tensor(list(y_dict.values())).to(torch.long)
        else:
            gnn_g.y = torch.tensor(graph_properties(gnn_g,targets)).to(torch.float)
            gnn_g.y_name=targets
    else:
        #G=igraph.Graph.to_networkx(g)
        gnn_g.y=None
    return gnn_g


def pre_DataSet_spdata( spdata_file_path,windows,observations,**params):
    from tqdm import tqdm
    import pickle
    dataSet = []
    print("loading adj")
    with open(spdata_file_path + '/As.pkl', "rb") as file:
        dataset_adj_list = pickle.load(file)
    print("loading ts")
    with open(spdata_file_path + '/numes.pkl', "rb") as file:
        dataset_ts_list = pickle.load(file)


    max_sample_num=params["max_sample_num"]
    max_degree_dataset = 0
    print("processing....")
    for adj, ts in tqdm(zip(dataset_adj_list, dataset_ts_list)):

        adj = torch.tensor(adj)
        # print(adj.shape)
        ts = torch.tensor(ts)
        ##print(ts.shape)

        if adj.shape[0] <= 20:  # smaller than 20 nodes, filtering..
            continue
        assert ts.shape[2] == adj.shape[0]
        assert ts.shape[1] >= 2*windows
        assert ts.shape[0] >= observations

        edge_index = tg_utils.dense_to_sparse(adj)[0]
        # print(edge_index.type())
        ts = ts.permute(2, 0, 1).unsqueeze(-1).float()  # [N,ob,windows,1]
        num_observations=ts.shape[1]
        # print(x.type())
        total_possible = math.comb(num_observations, observations)
        if total_possible <= max_sample_num:
            all_indices = list(it.combinations(range(num_observations), observations))
        else:
            # 随机采样指定数量的组合
            all_indices = set()
            while len(all_indices) < max_sample_num:
                # 随机选择k个不重复的索引
                random_sample = tuple(sorted(random.sample(range(num_observations), observations)))
                all_indices.add(random_sample)
            all_indices = list(all_indices)
        sample_count=0
        for indices in all_indices:
            ts_sample = ts[:, indices, :, :]
            x=ts_sample[:,:,:windows,:]
            y=ts_sample[:,:,windows:2*windows,:]
            # print(y.type())

            assert y.shape[2] == x.shape[2]
            pyg = Data(x=x, edge_index=edge_index, y=y)
            dataSet.append(pyg)
            sample_count=sample_count+1
            if sample_count>=max_sample_num:
                break

        tp_max_degree = tg_utils.degree(pyg.edge_index[0]).max().item()
        if tp_max_degree > max_degree_dataset:
            max_degree_dataset = tp_max_degree
        # if len(dataSet)>=100:
        #     break

    return dataSet, max_degree_dataset


def select_balanced_samples(pos_indices, neg_indices, sample_seed=233,num_per_class=10):
    # 检查是否有足够样本
    if len(pos_indices) < num_per_class:
        print(f"警告: 正样本数量不足{num_per_class}个，只有{len(pos_indices)}个")
        num_per_class = min(len(pos_indices), len(neg_indices))

    if len(neg_indices) < num_per_class:
        print(f"警告: 负样本数量不足{num_per_class}个，只有{len(neg_indices)}个")
        num_per_class = min(len(pos_indices), len(neg_indices))

    # 打乱正负样本索引
    rng = np.random.RandomState(sample_seed)
    shuffled_pos_indices = rng.permutation(pos_indices).tolist()
    shuffled_neg_indices = rng.permutation(neg_indices).tolist()


    # 各取前num_per_class个
    selected_pos = shuffled_pos_indices[:num_per_class]
    selected_neg = shuffled_neg_indices[:num_per_class]

    # 合并并打乱最终选择的样本索引
    selected_indices = selected_pos + selected_neg
    random.shuffle(selected_indices)

    return selected_indices, selected_pos, selected_neg

def select_balanced_samples_multi(sample_indices, sample_seed=233,num_per_class=10):
    # 检查是否有足够样本
    selected_indices=[]
    for cata,cata_indices in sample_indices.items():
        if len(cata_indices) < num_per_class:
            print(f"警告: {cata} 样本数量不足{num_per_class}个，只有{len(cata_indices)}个")
            num_per_class = min(len(cata_indices), num_per_class)



        # 打乱正负样本索引
        rng = np.random.RandomState(sample_seed)
        shuffled_pos_indices = rng.permutation(cata_indices).tolist()



        # 各取前num_per_class个
        selected_sample = shuffled_pos_indices[:num_per_class]


    # 合并并打乱最终选择的样本索引
        selected_indices = selected_indices + selected_sample
    random.shuffle(selected_indices)

    return selected_indices
# 分离正负样本索引
def get_label_indices(dataset):
    pos_indices = []
    neg_indices = []

    for idx in range(len(dataset)):
        pyg = dataset[idx]
        label=pyg.y
        if isinstance(label, torch.Tensor):
            label = label.item()

        if label == 1:  # 正样本
            pos_indices.append(idx)
        elif label == 0:  # 负样本
            neg_indices.append(idx)

    return pos_indices, neg_indices
def get_label_indices_multi(dataset,cata_num=3):
    sample_indices = {}
    for category in range(cata_num):
        sample_indices[category]=[]

    for idx in range(len(dataset)):
        pyg = dataset[idx]
        label=pyg.y
        if isinstance(label, torch.Tensor):
            label = label.item()
        assert label <cata_num
        sample_indices[label].append(idx)

    return sample_indices
def pred_DataSet_supervised( spdata_file_path,windows,observations,**params):
    from tqdm import tqdm
    import pickle
    dataSet = []
    print("loading adj")
    with open(spdata_file_path + '/As.pkl', "rb") as file:
        dataset_adj_list = pickle.load(file)
    print("loading ts")
    with open(spdata_file_path + '/numes.pkl', "rb") as file:
        dataset_ts_list = pickle.load(file)
    with open(spdata_file_path + '/rs.pkl', "rb") as file:
        dataset_label_list = pickle.load(file)


    max_sample_num=params["max_sample_num"]

    print("processing....")
    for adj, ts,lable in tqdm(zip(dataset_adj_list, dataset_ts_list,dataset_label_list)):
        num_nodes=adj.shape[0]
        adj = torch.tensor(adj)
        # print(adj.shape)
        ts = torch.tensor(ts)
        lable=torch.tensor(lable).unsqueeze(-1)
        #print(ts.shape)
        #print(lable.shape)

        if adj.shape[0] <= 20:  # smaller than 20 nodes, filtering..
            continue
        assert ts.shape[2] == adj.shape[0]
        assert ts.shape[1] >= 2*windows
        assert ts.shape[0] >= observations

        edge_index = tg_utils.dense_to_sparse(adj)[0]
        # print(edge_index.type())
        ts = ts.permute(2, 0, 1).unsqueeze(-1).float()  # [N,ob,windows,1]
        num_observations=ts.shape[1]
        # print(x.type())
        total_possible = math.comb(num_observations, observations)
        if total_possible <= max_sample_num:
            all_indices = list(it.combinations(range(num_observations), observations))
        else:
            # 随机采样指定数量的组合
            all_indices = set()
            while len(all_indices) < max_sample_num:
                # 随机选择k个不重复的索引
                random_sample = tuple(sorted(random.sample(range(num_observations), observations)))
                all_indices.add(random_sample)
            all_indices = list(all_indices)
        sample_count=0
        for indices in all_indices:
            ts_sample = ts[:, indices, :, :]
            x=ts_sample[:,:,:windows,:]
            y=lable

            # print(y.type())


            pyg = Data(x=x, edge_index=edge_index, y=y,num_nodes=num_nodes)
            dataSet.append(pyg)
            sample_count=sample_count+1
            if sample_count>=max_sample_num:
                break



        # if len(dataSet)>=100:
        #     break

    return dataSet
def normalized_laplacian(A):
    """
    Input A: np.ndarray
    :return:  np.ndarray  D^-1/2 * ( D - A ) * D^-1/2 = I - D^-1/2 * ( A ) * D^-1/2
    """
    out_degree = np.array(A.sum(1), dtype=np.float32)
    int_degree = np.array(A.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.eye(A.shape[0]) - np.diag(out_degree_sqrt_inv) @ A @ np.diag(int_degree_sqrt_inv)
    return mx_operator

def zipf_smoothing(A):
    """
    Input A: np.ndarray
    :return:  np.ndarray  (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    """
    A_prime = A + np.eye(A.shape[0])
    out_degree = np.array(A_prime.sum(1), dtype=np.float32)
    int_degree = np.array(A_prime.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.diag(out_degree_sqrt_inv) @ A_prime @ np.diag(int_degree_sqrt_inv)
    return mx_operator

def normalized_adj(A):
    """
    Input A: np.ndarray
    :return:  np.ndarray  D^-1/2 *  A   * D^-1/2
    """
    out_degree = np.array(A.sum(1), dtype=np.float32)
    int_degree = np.array(A.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.diag(out_degree_sqrt_inv) @ A @ np.diag(int_degree_sqrt_inv)
    return mx_operator

def process_instance(gdata, device):
    A=tg_utils.to_dense_adj(gdata.edge_index,max_num_nodes=gdata.num_nodes).cpu().detach().numpy()
    A = np.array(A[0])
   # print("A",A.shape)

    add0 = np.ones((1, A.shape[0]))
    add1 = np.zeros((A.shape[0]+1, 1))
    A = np.concatenate((A, add0), axis=0)
    A = np.concatenate((A, add1), axis=1)
    assert A.shape[0] == A.shape[1]

    A = normalized_adj(A)


    A = torch.from_numpy(A).to(torch.float32).to(device)
    numericals=gdata.x.cpu().detach().squeeze(-1).numpy()# # [N,ob,windows]

    numericals = np.transpose(numericals, (1,0,2))#[ob,N,windows]
  #  print("numericals",numericals.shape)
    add_nume = np.mean(numericals, axis=1, keepdims=True)
    numericals = np.concatenate((numericals, add_nume), axis=1)
    numericals = torch.from_numpy(numericals).to(torch.float32).to(device)
    r_truth = gdata.y.to(torch.float32).to(device)

    return A, numericals, r_truth
def draw_3d(G,ax,pos,color_value):

    x, y, z = zip(*pos.values())

    for n, m in G.edges():
        zline = (z[n], z[m])
        xline = (x[n], x[m])
        yline = (y[n], y[m])
        ax.plot3D(xline, yline, zline,'k',linewidth=1)
    return ax.scatter3D(x, y, z, c=color_value, marker='o',alpha=1, s=100)

def visualization_evalution(pred,graph,target):


    data = graph  #可随机选取
    out =sigmoid( pred)
    G=torch_geometric.utils.to_networkx(data,to_undirected=True)

    ###########################2D可视化展示（可选）###########################################################
    fig=plt.figure("2d visualization of predict")
    pos = nx.kamada_kawai_layout(G)

    ax=fig.add_subplot(211)
    ax.set(title="Prediction  using GAT")
    nodes = nx.draw_networkx_nodes(G,pos=pos, node_color=out)
    nx.draw_networkx_edges(G, pos=pos,width=1)
    fig.colorbar(nodes)

    ax = fig.add_subplot(212)
    ax.set(title="Labels of network key nodes ")
    nodes = nx.draw_networkx_nodes(G, pos=pos, node_color=target)
    nx.draw_networkx_edges(G, pos=pos, width=1)
    fig.colorbar(nodes)
    #######################################################################################################
    ###########################3D可视化展示（可选）###########################################################
    # fig = plt.figure("3d visualization of predict")
    # pos = nx.kamada_kawai_layout(G,dim=3)
    #
    # ax = fig.add_subplot(121,projection="3d")
    # ax.set(title="Prediction  using GAT")
    # ax0=draw_3d(G=G,pos=pos,ax=ax,color_value=out)
    # fig.colorbar(ax0,ax=ax)
    #
    # ax = fig.add_subplot(122,projection="3d")
    # ax.set(title="Labels of network key nodes ")
    # ax0 = draw_3d(G=G, pos=pos, ax=ax, color_value=target)
    # fig.colorbar(ax0, ax=ax)
    #######################################################################################################




    return 0
def pred_accuracy(pred, y, num_graph,data):  # torch tensor变量 可batch
    sum=0
    last_split=0
    for i in range(num_graph):
        batch_node = data[i].num_nodes
        c_num=math.ceil(batch_node*0.6)
        out_node = pred[last_split: last_split + batch_node]
        label_node = y[last_split: last_split + batch_node]
        out_rank = np.argsort(out_node, axis=0).reshape(-1)
        label_rank = np.argsort(label_node, axis=0).reshape(-1)
        correct = (out_rank[:c_num] == label_rank[:c_num])
        accuracy = int(correct.sum()) / correct.numel()
        sum += accuracy
        last_split = last_split + batch_node

    return sum/num_graph
def kendall_rank_coffecient(out, label, num_graph, data):
    sum = 0
    last_split = 0
    for i in range(num_graph):
        batch_node = data[i].num_nodes
        out_node = out[last_split: last_split + batch_node]
        label_node = label[last_split: last_split + batch_node]
      #  out_rank = np.argsort(out_node, axis=0).reshape(-1)
      #  label_rank = np.argsort(label_node, axis=0).reshape(-1)
        tau, p_value = stats.kendalltau(out_node, label_node)
        sum += tau
        last_split = last_split + batch_node

    return sum/num_graph
def set_correlation_coffecient(out,label):
    pred_set = set(np.where(out == 1)[0].tolist())
    label_set=set(np.where(label == 1)[0].tolist())
    intersection_set_len = len(pred_set & label_set)
    union_set_len = len(pred_set | label_set)
    # difference_set_len=len(set_1-set_2)

    correlation_coefficient = (intersection_set_len) / union_set_len
    return correlation_coefficient
def visualize_node_emb(fig_label,node_emb,node_mean,node_var,target):
    z = TSNE(n_components=1).fit_transform(node_emb)
    # mean=sigmoid(node_mean)
    # var=sigmoid(node_var+node_mean)-mean
    mean=node_mean
    var=node_var
    print("mean:{}".format(node_mean))
    print("var:{}".format(var))
    print("target:{}".format(target))

    fig=plt.figure("3D visualization of  "+fig_label)
   # ax=fig.add_subplot(111,projection="3d")
    # ax.scatter(z[:, 0], z[:, 1], mean.T,   c="k",marker='o', alpha=1, s=10)
    # ax.scatter(z[:, 0], z[:, 1], target.T, c="r", marker='o', alpha=1, s=10)
    # ax.scatter(z[:, 0], z[:, 1], mean.T,   c=mean,    marker='o', alpha=0.7, s=10*var)
    ax = fig.add_subplot(111)
    ax.scatter(z[:, 0], mean.T, c="k", marker='o', alpha=1, s=10)
    ax.scatter(z[:, 0], target.T, c="r", marker='o', alpha=1, s=10)
    ax.scatter(z[:, 0],  mean.T, c='b', marker='o', alpha=0.5, s=10+10 * var)

def save_checkpoint(path: str,model_name: str, model,net_param):
    """
    Saves a model checkpoint.
    """
    # Convert args to namespace for backwards compatibility

    model_state = {
        'net_param': net_param,
        'state_dict': model.state_dict(),
    }
    model_path=os.path.join(path,model_name)
    torch.save(model_state,model_path)


def load_Resilience_model(path: str, device,load_state=True, infer_para=None, **kwargs) :
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :return: The loaded model,loaded network parameters.
    """

    # Load model and args
    with open(path, 'rb') as f:
        state = torch.load(f, map_location=lambda storage, loc: storage)

    loaded_net_param = state["net_param"]
   # print(loaded_net_param)
    if infer_para is not None:
        loaded_net_param.update(infer_para)
    loaded_state_dict = state['state_dict']
    if not torch.cuda.device_count() > 1:
       # print(loaded_state_dict.keys())
        loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
   # print("loaded_net_param:{}".format(loaded_state_dict))

    loaded_net_param["device"] = device

    model = Resilience_model( net_param=loaded_net_param).to(
        device)
    if load_state:
        model.load_state_dict(loaded_state_dict,strict=True)
    model = model.to(device)

    return model,loaded_net_param


def emergency_checkpoint(model, net_param,optimizer, scheduler, step, record_scores, checkpoint_path='emergency_checkpoint.pth'):
    checkpoint = {
        'step': step,
        "record_scores": record_scores,
        "mdoel_params": net_param,
        'model_state_dict': model.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
    }
    if isinstance(optimizer,dict):
        checkpoint["optimizer_state_dict"]=dict()
        for name in optimizer.keys():
            checkpoint["optimizer_state_dict"][name]=optimizer[name].state_dict()
    else:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    # 尝试先保存到临时文件，再替换，以避免写入中断导致文件损坏
    checkpoint_path=checkpoint_path+"/emergency_checkpoint.pth"
    temp_path = checkpoint_path + '.tmp'

    torch.save(checkpoint, temp_path)
    os.replace(temp_path, checkpoint_path)
    print(f"紧急检查点已保存至: {checkpoint_path}")
def load_emergency_checkpoint_multi(checkpoint_path, model, optimizer,device, scheduler=None,score_metrics=None):
    temp_path = checkpoint_path + '/emergency_checkpoint.pth'
    if not os.path.exists(temp_path):
      #  score_metrics = ["predictability", "reconstructability", "multitask"]
        record_scores = {
            "epoch": list(),

        }
        if score_metrics is not None:
            for score_metric in score_metrics:
                record_scores[score_metric]={"train_scores": list(),
                "val_scores": list(),}
        else:
            record_scores["train_scores"]=list()
            record_scores["val_scores"]=list()
        return 0, record_scores
    else:
        print(f"加载紧急检查点: {temp_path}")
        with open(temp_path, 'rb') as f:
            checkpoint = torch.load(f, map_location=device)  # 先加载到CPU
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        if not isinstance(optimizer,dict):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            for name in optimizer.keys():
                optimizer[name].load_state_dict(checkpoint['optimizer_state_dict'][name])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'],strict=True)
        return checkpoint['step'],checkpoint['record_scores']

def load_emergency_checkpoint(checkpoint_path, model, optimizer,device, scheduler=None):
    temp_path = checkpoint_path + '/emergency_checkpoint.pth'
    if not os.path.exists(temp_path):

        record_scores = {
            "epoch": list(),
            "train_scores": list(),
            "val_scores": list()
        }

        return 0, record_scores
    else:
        print(f"加载紧急检查点: {temp_path}")
        with open(temp_path, 'rb') as f:
            checkpoint = torch.load(f, map_location=device)  # 先加载到CPU
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'],strict=True)
        return checkpoint['step'],checkpoint['record_scores']
def save_config(path:str,configs_name="configs.yaml",dataset_param=None,net_param=None,train_param=None,optimizer_param=None,loss_param=None):
    train_state = {
        'dataset':dataset_param,
        'train': train_param,
        'net': net_param,
        'optimizer': optimizer_param,
        'loss': loss_param,
    }

    file_path=os.path.join(path,configs_name)


    if os.path.exists(file_path):

        with open(file_path, 'r') as f:
            saved_train_parameters=yaml.safe_load(f)

        if json.dumps(saved_train_parameters,sort_keys=True) ==json.dumps(train_state,sort_keys=True):
            trained_model_path = os.path.join(path, "hold_out/trained_model")
            #print(trained_model_path)
            if os.path.exists(trained_model_path):
                configs_name=configs_name.replace("config_", "")
                print("{} model has existed".format(configs_name))
                return False
            else:
                return True
        else:
            with open(file_path, "w") as f:
                yaml.dump(train_state, f)
            return True
    else:
        with open(file_path, "w") as f:
            yaml.dump(train_state, f)
        return True

if __name__=="__main__":
    from torch_geometric.loader import DataLoader
    # dataset_param={
    #
    #                "features": "none",
    #                "filter":"*",
    #                "SL":False,
    #                 "data_num":100,
    #                 "g_types":['erdos_renyi','small-world','barabasi_albert',"static_power_law","K_Regular"]
    #                }
    #
    # trainset=pre_DataSet(**dataset_param)
    # train_loader = DataLoader(trainset, batch_size=8, shuffle=True, drop_last=False)
    # for n, data in enumerate(train_loader):
    #     torch_graphs=data.to_data_list()
    #     for torch_graph in torch_graphs:
    #         predata = torch_graph.y[0]
    #         print(type(predata))
