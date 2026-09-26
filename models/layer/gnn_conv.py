import torch_geometric.nn as gnn
from .AGNNConv import AGNNConv
from .FGNNConv import FGNNConv
def gnn_conv(gnn_name,in_channels,out_channels,gnn_param:dict={}):
    if gnn_name=="GATConv" :
        return gnn.GATConv(in_channels=in_channels,out_channels=out_channels,**gnn_param)
    elif gnn_name=="GATv2Conv" :
        return gnn.GATv2Conv(in_channels=in_channels,out_channels=out_channels,**gnn_param)
    elif gnn_name=="GCNConv":
        return gnn.GCNConv(in_channels=in_channels,out_channels=out_channels,**gnn_param)
    elif gnn_name=="SAGEConv":
        return gnn.SAGEConv(in_channels=in_channels,out_channels=out_channels,**gnn_param)
    elif gnn_name=="GraphConv":
        return gnn.GraphConv(in_channels=in_channels,out_channels=out_channels,**gnn_param)
    elif gnn_name=="GatedGraphConv":
        return gnn.GatedGraphConv(out_channels=out_channels,**gnn_param)
    elif gnn_name=="ResGatedGraphConv":
        return gnn.ResGatedGraphConv(in_channels=in_channels,out_channels=out_channels,**gnn_param)
    elif gnn_name=="PNAConv":
        return gnn.PNAConv(in_channels=in_channels,out_channels=out_channels,**gnn_param)
    elif gnn_name=="AGNNConv":
        return AGNNConv(in_channels=in_channels,out_channels=out_channels,**gnn_param)
    elif gnn_name=="FGNNConv":
        return FGNNConv(input_vdim=in_channels,output_vdim=out_channels,**gnn_param)
    else:
        raise ValueError("the definition  don't exit\n"
                         "\tyou can define it before using it")