import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader

from torch_geometric.nn import MessagePassing
import torch_geometric.utils as tg_utils


class QNetwork(nn.Module):
    def __init__(self, edge_dim, k, l,num_layers):
        super(QNetwork, self).__init__()
        self.k = k
        self.l = l
        self.fc = nn.Sequential()
        for i in range(num_layers - 1):
            if i == 0:
                self.fc.append(nn.Linear(edge_dim, self.k * self.l))
            else:
                self.fc.append(nn.Linear(self.k * self.l, self.k * self.l))
            self.fc.append(nn.ReLU())
        if num_layers == 1:
            self.fc.append(nn.Linear(edge_dim, self.k * self.l))
        else:
            self.fc.append(nn.Linear(self.k * self.l, self.k * self.l))

    def forward(self, edge_features):
        return self.fc(edge_features).view(-1, self.k, self.l)


class MNetwork(nn.Module):
    def __init__(self, input_dim, output_dim,num_layers):
        super(MNetwork, self).__init__()
        self.fc = nn.Sequential()
        for i in range(num_layers - 1):
            if i == 0:
                self.fc.append(nn.Linear(input_dim, output_dim))
            else:
                self.fc.append(nn.Linear(output_dim, output_dim))
            self.fc.append(nn.ReLU())
        if num_layers == 1:
            self.fc.append(nn.Linear(input_dim, output_dim))
        else:
            self.fc.append(nn.Linear(output_dim, output_dim))

    def forward(self, node_features):
        return self.fc(node_features)


class FGNNConv(MessagePassing):
    """
    Factor Graph Neural Network (FGNN) convolution layer.
    """
    def __init__(self, input_vdim, output_vdim,factor_dim,edge_dim,net_num_layers,aggr='sum'):
        super(FGNNConv, self).__init__(aggr=aggr)
        """
        input_vdim: dimension of the input variable node features
        output_vdim: dimension of the output variable node features
        factor_dim: dimension of the factor node features
        edge_dim: dimension of the edge features
        net_num_layers: number of layers in the Q and M networks
        aggr: aggregation method for message passing (default: sum) , options: mean, max, min, sum, mul
        """

        # Define variable-to-factor  Q and M networks
        self.Q_vf = QNetwork(edge_dim, factor_dim, output_vdim,net_num_layers)
        self.M_vf = MNetwork(input_vdim + factor_dim, output_vdim,net_num_layers)
        # Define factor-to-variable  Q and M networks
        self.Q_fv = QNetwork(edge_dim, output_vdim, factor_dim,     net_num_layers)
        self.M_fv = MNetwork(input_vdim + factor_dim, factor_dim,net_num_layers)

    def forward(self, var_features, factor_features,
                v2f_edge_index, edge_attr,
                f2v_edge_index):
        # Variable-to-Factor
        var_nodes=int(var_features.shape[0])
        factor_nodes=int(factor_features.shape[0])

        out_vf = self.propagate(v2f_edge_index, x=(var_features, factor_features), edge_attr=edge_attr, method='vf',size=[var_nodes,factor_nodes])

        # Factor-to-Variable
        out_fv = self.propagate(f2v_edge_index, x=(factor_features, var_features), edge_attr=edge_attr, method='fv',size=[factor_nodes,var_nodes])

        return  out_fv, out_vf,edge_attr

    def message(self, x_i, x_j, edge_attr, method):

        if method == 'vf':
            q = self.Q_vf(edge_attr)
            m = self.M_vf(torch.cat([x_i, x_j], dim=-1))
        elif method == 'fv':
            q = self.Q_fv(edge_attr)
            m = self.M_fv(torch.cat([x_i, x_j], dim=-1))

        return torch.bmm(q, m.unsqueeze(-1)).squeeze(-1)


def find_cliques(data):
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    triangles = []

    # 构建邻居列表
    neighbors = [[] for _ in range(num_nodes)]
    for i, j in edge_index.t().tolist():
        neighbors[i].append(j)
        neighbors[j].append(i)

    # 寻找三角形团
    for node in range(num_nodes):
        node_neighbors = sorted(neighbors[node])
        for i in range(len(node_neighbors)):
            for j in range(i + 1, len(node_neighbors)):
                u, v = node_neighbors[i], node_neighbors[j]
                # 检查共同邻居
                if v in neighbors[u]:
                    triangle = {node, u, v}
                    if triangle not in triangles:
                        triangles.append(triangle)
    return triangles


def bron_kerbosch(R, P, X, neighbors, cliques):
    if not P and not X:
        cliques.append(R)
    while P:
        v = P.pop()
        new_R = R | {v}
        new_P = P & neighbors[v]
        new_X = X & neighbors[v]
        bron_kerbosch(new_R, new_P, new_X, neighbors, cliques)
        X.add(v)


def find_maximal_cliques(data):
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    neighbors = [set() for _ in range(num_nodes)]

    for i, j in edge_index.t().tolist():
        neighbors[i].add(j)
        neighbors[j].add(i)

    cliques = []
    bron_kerbosch(set(), set(range(num_nodes)), set(), neighbors, cliques)
    return cliques


def bron_kerbosch_with_limit(R, P, X, neighbors, max_clique_size, cliques):
    """
        改进的 Bron-Kerbosch 算法，支持最大团簇规模限制，使用集合加速操作。

        参数：
        - R: 当前团簇（集合，正在构建的子图）。
        - P: 候选节点集合（集合）。
        - X: 已处理的节点集合（集合）。
        - neighbors: 邻接表。
        - max_clique_size: 最大团簇规模限制。
        - cliques: 存储找到的团簇的集合。
        """
    if len(R) > max_clique_size:
        # 如果当前团簇超过规模限制，将其拆分为符合条件的小团簇
        cliques.update(split_clique(R, max_clique_size))  # 使用集合的更新操作
        return

    if not P and not X:
        # 如果没有候选节点和已处理节点，当前团簇是极大团簇
        cliques.add(frozenset(R))  # 使用 frozenset 确保集合可哈希
        return

    # 遍历候选节点 P 中的每个节点
    while P:
        v = P.pop()  # 从 P 中选择一个节点
        new_R = R | {v}  # 将 v 加入当前团簇
        new_P = P & neighbors[v]  # 计算与 v 相连的候选节点
        new_X = X & neighbors[v]  # 计算与 v 相连的已处理节点
        bron_kerbosch_with_limit(new_R, new_P, new_X, neighbors, max_clique_size, cliques)
        X.add(v)  # 将 v 移动到已处理节点集合


def split_clique(clique, max_clique_size):
    """
    将超出规模限制的团簇拆分为多个符合最大规模限制的小团簇。

    参数：
    - clique: 超出规模限制的团簇节点集合。
    - max_clique_size: 最大团簇规模限制。

    返回：
    - 小团簇列表，每个团簇不超过 max_clique_size。
    """
    from itertools import combinations
    # 枚举节点组合，确保每个组合是完全子图
    return {frozenset(sub_clique) for sub_clique in combinations(clique, max_clique_size)}


def find_cliques_with_limit(data, max_clique_size):
    """
    使用改进的 Bron-Kerbosch 算法寻找满足规模限制的团簇。

    参数：
    - data: torch_geometric.data.Data 图数据对象，包含 edge_index 和 num_nodes。
    - max_clique_size: 最大团簇规模限制。

    返回：
    - cliques: 找到的所有团簇列表。
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    # 构建邻接表
    neighbors = {i: set() for i in range(num_nodes)}
    for u, v in edge_index.t().tolist():
        neighbors[u].add(v)
        neighbors[v].add(u)  # 无向图需要双向添加

    cliques = set()  # 使用集合存储团簇
    bron_kerbosch_with_limit(set(), set(range(num_nodes)), set(), neighbors, max_clique_size, cliques)
    return cliques

def convert_simple_graph_to_factor_graph(orig_data,factor_dim,edge_dim,higher_order=None,edge_attr=None):
    """
    Create a factor graph dataset using HeteroData from a simple graph with undirected edges and node features.

    Parameters:
    - orig_data: Data object containing the simple graph with node features and edge list
    - factor_dim: Dimension of the factor node features
    - edge_dim: Dimension of the edge features
    Returns:
    - data: HeteroData object containing all information of the factor graph
    """
    # Create HeteroData object
    if not tg_utils.is_undirected(orig_data.edge_index):

        raise ValueError("Simple graph must be undirected")
    num_edges = orig_data.num_edges
    edge_index = orig_data.edge_index
    node_features = orig_data.x
    data = HeteroData()

    # Add variable node features
    data['variable'].x = node_features
    data['variable'].y = orig_data.y
    data['variable', 'to', 'factor'].src4v_edge_index = orig_data.edge_index
    data['variable', 'to', 'factor'].tgt4v_edge_index = torch.flip(orig_data.edge_index, dims=[0])




    # Add connection relationships
    # Add edges from variable nodes to factor nodes
    factor_edge = []
    if higher_order is None:
        cliques = find_maximal_cliques(orig_data)
    else:
        cliques = find_cliques_with_limit(orig_data,max_clique_size=(higher_order+1))
    factor_node_features = torch.zeros(len(cliques), factor_dim)  # features for factor nodes
    data['factor'].x = factor_node_features
    for factor_node, clique in enumerate(cliques):
        for var_node in clique:
            factor_edge.append([var_node, factor_node])

    factor_edge_index=torch.tensor(factor_edge,dtype=torch.long).t()

    data['variable', 'to', 'factor'].edge_index = factor_edge_index

    # Add edge attributes
    if edge_attr is None:
        data['variable', 'to', 'factor'].edge_attr = torch.ones(data['variable', 'to', 'factor'].edge_index.size(1), edge_dim)
        data['variable', 'to', 'factor'].edge_attr = edge_attr

    return data


if __name__ == '__main__':
    class FGNN(nn.Module):
        def __init__(self, var_dim, factor_dim, edge_dim, k,  num_layers):
            super(FGNN, self).__init__()
            self.layers = nn.ModuleList()

            for n in range(2):
                if n == 0:
                    self.layers.append(FGNNConv(input_vdim=var_dim, output_vdim=k, factor_dim=factor_dim, edge_dim=edge_dim,net_num_layers=num_layers))
                else:
                    self.layers.append(FGNNConv(input_vdim=k, output_vdim=k, factor_dim=factor_dim, edge_dim=edge_dim,net_num_layers=num_layers))

        def forward(self,factor_graph_data):
            var_features, factor_features = factor_graph_data["variable"].x, factor_graph_data["factor"].x
            v2f_edge_index, edge_attr = factor_graph_data["variable", "to", "factor"].edge_index, factor_graph_data["variable", "to", "factor"].edge_attr
            f2v_edge_index = torch.flip(v2f_edge_index,dims=[0])

            for layer in self.layers:
                v_in,f_in = var_features,factor_features
                var_features, factor_features,_ = layer(var_features, factor_features,
                                                      v2f_edge_index, edge_attr,
                                                       f2v_edge_index)
                var_features = F.normalize(input=var_features)
                factor_features = F.normalize(input=factor_features)
                # activation
                var_features = F.relu(var_features)
                factor_features = F.relu(factor_features)
                #residual connection
                var_features=var_features+v_in
                factor_features=factor_features+f_in
            return var_features, factor_features


    def duplicate_edge_index( parallel_sampling, edge_index, v_num_nodes, f_num_nodes, device):
        """Duplicate the edge index (in sparse graphs) for parallel sampling."""
        v_edge_index = edge_index[0].reshape((1, 1, -1))
        f_edge_index = edge_index[1].reshape((1, 1, -1))
        edge_index_indent = torch.arange(0, parallel_sampling).view(1, -1, 1).to(device)
        v_edge_index_indent = edge_index_indent * v_num_nodes
        f_edge_index_indent = edge_index_indent * f_num_nodes

        v_edge_index_indent = v_edge_index + v_edge_index_indent
        f_edge_index_indent = f_edge_index + f_edge_index_indent
        edge_index = torch.cat((v_edge_index_indent.reshape((-1, 1)), f_edge_index_indent.reshape((-1, 1))), dim=0)
        edge_index = edge_index.reshape((2, -1))
        return edge_index
    #
    #
    # # Example usage

   #  factor_dim = 20
   #  edge_dim = 20
   #  k = 20
   #
   #  num_layers = 2
   #  graph_tensor_save_path = "analysis/analysis_dataset/dataset_tensor/cfinder-google.pt"
   #  print("数据加载")
   #  factor_graph = torch.load(graph_tensor_save_path).to('cpu')
   #
   #  var_dim = factor_graph["variable"].x.shape[1]
   #
   #  fgnn = FGNN(var_dim, factor_dim, edge_dim, k,  num_layers).to('cpu')
   #
   #  #
   #  # # Create a graph with variable and factor nodes
   #  # edge_index = torch.tensor([[0, 1, 2, 3],
   #  #                            [0, 0, 1, 1]] , dtype=torch.long) # Variable to factor edges
   #  #
   #  # var_features = torch.randn(4, var_dim)  # 4 variable nodes
   #  # factor_features = torch.randn(2, factor_dim)  # 2 factor nodes
   #  # edge_attr = torch.randn(edge_index.size(1), edge_dim)  # Edge features
   #  #
   #
   #
   # #  # Example simple graph with edge list and node features
   # #  edge_index = torch.tensor([[0,0, 1, 1, 2,3],  # Source nodes
   # #                             [1,3, 0, 2, 1,0]], dtype=torch.long)  # Target nodes
   # # # [[0, 1, 1, 2],
   # #  # [0, 0, 1, 1],]
   # #
   # #  node_features = torch.randn(4, var_dim)  # 4 nodes with feature dimension 10
   # #  data = Data(x=node_features, edge_index=edge_index)
   # #  # Convert simple graph to factor graph
   # #  factor_graph = convert_simple_graph_to_factor_graph(data,factor_dim,edge_dim)
   # #
   # #  # Print the resulting factor graph
   # #  print(factor_graph)
   # #  print( factor_graph["variable", "to", "factor"].edge_index)
   # #  data_list = [factor_graph, factor_graph]
   # #
   # #  # 使用DataLoader进行批处理
   # #  loader = DataLoader(data_list, batch_size=2)
   # #
   # #  # 获取一批数据
   # #  for n,batch in enumerate(loader):
   # #      print(batch)
   # #
   # #      print("source for v edge index:",batch["variable", "to", "factor"].src4v_edge_index   )
   # #      print("target for v edge index:",batch["variable", "to", "factor"].tgt4v_edge_index   )
   # #      print("v2f edge index:",batch["variable", "to", "factor"].edge_index   )
   #  #   print("f2v edge index:",batch["variable", "to", "factor"].edge_index   )
   #  output_var, output_factor = fgnn(factor_graph)
   #
   #  print(output_var)
   #  print(output_factor)

    edge_index = torch.tensor([[0, 1, 2, 3, 0, 3,1,3,0,2,1,2,1,4,4,4,4,2,0,3],
                               [1, 0, 3, 2, 3, 0,3,1,2,0,2,1,4,1,0,2,3,4,4,4]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1], [2],[3]], dtype=torch.float)  # 节点特征

    data = Data(x=x, edge_index=edge_index)
  #  factor_graph = convert_simple_graph_to_factor_graph(data, factor_dim=10, edge_dim=10, higher_order=True)
    # 寻找团
    cliques = find_maximal_cliques(data)
    print(f"共找到 {len(cliques)} 个团")
    # 输出每个团
    for i, clique in enumerate(cliques):
        print(type(clique))
        print(f"团 {i + 1}: {sorted(clique)}")
    size_limit = 4

    cliques = find_cliques_with_limit(data,max_clique_size = size_limit)
    print(f"共找到 {len(cliques)} 个团")
    # 输出每个团
    for i, clique in enumerate(cliques):
        print(type(clique))
        print(f"团 {i + 1}: {clique}")
        for var_node in clique:
            print(type(var_node))

   # print("factor graph:",factor_graph)
   # print("variable to factor edge index:",factor_graph["variable", "to", "factor"].edge_index)