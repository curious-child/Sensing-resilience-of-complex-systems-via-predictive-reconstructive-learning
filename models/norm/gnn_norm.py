import torch_geometric.nn as gnn
import torch
import torch.nn as nn
import torch_geometric as tg
def gnn_norm(norm_name,in_channerls):
    if norm_name=="GraphNorm":
        return gnn.GraphNorm(in_channels=in_channerls)
    elif norm_name=="GraphSizeNorm":
        return gnn.GraphSizeNorm()
    elif norm_name=="MeanSubtractionNorm":
        return gnn.MeanSubtractionNorm()
    elif norm_name=="PairNorm":
        return gnn.PairNorm()
    elif norm_name=="BatchNorm":
        return gnn.norm.BatchNorm(in_channels=in_channerls)
    else:
        raise ValueError("the definition  don't exit\n"
                         "\tyou can define it before using it")