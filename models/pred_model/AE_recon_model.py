from types import SimpleNamespace
from easydict import EasyDict
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.utils import batched_negative_sampling, negative_sampling, to_dense_adj
import torch_geometric.utils as pyg_utils
from models.pool.gnn_pool import SelfAttentionPooling

from models.pred_model.NS_Transformer import ns_Transformer
import torch.nn.functional as F
EPS = 1e-15
MAX_LOGSTD = 10
class Hadamard_MLPPredictor(nn.Module):
    def __init__(self, h_feats, dropout=0.01, layer=2, res=True, norm=True, scale=False, act='relu'):
        super().__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(h_feats, h_feats))
        for _ in range(layer - 2):
            self.lins.append(torch.nn.Linear(h_feats, h_feats))
        self.lins.append(torch.nn.Linear(h_feats, 1))
        self.dropout = dropout
        self.res = res
        self.scale = scale
        if scale:
            self.scale_norm = nn.LayerNorm(2*h_feats)
        self.norm = norm
        if norm:
            self.norms = torch.nn.ModuleList()
            for _ in range(layer - 1):
                self.norms.append(nn.LayerNorm(h_feats))
        if act == 'relu':
            self.act = F.relu
        elif act == 'gelu':
            self.act = F.gelu
        elif act == 'silu':
            self.act = F.silu
        else:
            raise ValueError('Activation function not supported')

    def forward(self, z,edge_index,sigmoid = True):
        x_i=z[edge_index[0]]
        x_j=z[edge_index[1]]

        x = torch.cat([x_i , x_j], dim=-1)
        if self.scale:
            x = self.scale_norm(x)
        ori = x
        for i in range(len(self.lins) - 1):
            x = self.lins[i](x)
            if self.res:
                x += ori
            if self.norm:
                x = self.norms[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        value = self.lins[-1](x).squeeze()
        return torch.sigmoid(value) if sigmoid else value



class AE_recon_model(nn.Module):
    def __init__(self, net_param):
        super().__init__()
        self.log_var = net_param['log_var'] if net_param.get('log_var') else 0.0

        self.edge_decoder_type = net_param['edge_decoder_type']
        self.hidden_dim = net_param['hidden_dim']
        net_param["Dy_transformer"]['d_model']=self.hidden_dim
        self.device = net_param['device']
        self.neg_samples_ratio=net_param['neg_samples_ratio']


        self.degree_loss_weight=net_param["degree_loss_weight"]
        self.lr=net_param["lr"] if net_param.get('lr') else 0.001
        self.dataset_nf = net_param["Dy_transformer"]['dataset_nf']
        self.windows = net_param["Dy_transformer"]['windows']
        self.pred_len = net_param["Dy_transformer"]['pred_len'] =self.windows
        self.seq_len = net_param["Dy_transformer"]["seq_len"] = self.windows
        self.label_len = net_param["Dy_transformer"]["label_len"] = self.windows // 2
        net_param["Dy_transformer"]["device"]=self.device
        self.dy_configs = EasyDict(net_param["Dy_transformer"])
        self.cond_pred_model = ns_Transformer(self.dy_configs).float().to(self.device)
        self.pred_y_loss_weight = nn.Parameter(torch.tensor(self.log_var).float())
        self.traj_reduc= SelfAttentionPooling(emd_dim=self.hidden_dim)

        if self.edge_decoder_type == 'InnerProductDecoder':
            self.edge_decoder=InnerProductDecoder().to(self.device)
        elif self.edge_decoder_type == 'Hadamard_MLPPredictor':
            self.edge_decoder = Hadamard_MLPPredictor(2*self.hidden_dim).to(self.device)
        else:
            raise ValueError('Edge decoder type not supported')
        self.recon_loss_weight = nn.Parameter(torch.tensor(self.log_var).float())

    def get_representation_dim(self):
        return self.hidden_dim
    def training_step(self,gdata,task_cond=None):

        batch_index=gdata.batch.to(self.device)
        edge_index=gdata.edge_index.to(self.device)
        assert pyg_utils.is_undirected(edge_index),"currently only supports undirected graphs!!"
        N,M,W,F=gdata.x.shape
        assert F==self.dataset_nf
        assert W==self.windows
        o_batch_x=gdata.x #[N,M,window_size,dataset_nf]#M为观测轨迹个数
        flatten_batch_x=o_batch_x.reshape(N*M,self.windows,self.dataset_nf)
        batch_y=gdata.y #[N,M,window_size,dataset_nf]
        flatten_batch_y=batch_y.reshape(N*M,-1,self.dataset_nf)#[N*M,pred_len,dataset_nf]
        assert flatten_batch_x.shape[1] == self.windows
        assert flatten_batch_y.shape[1] == self.windows
        dec_inp_pred = torch.zeros(
            [flatten_batch_x.size(0), self.pred_len, self.dataset_nf]
        ).to(self.device)
        dec_inp_label = flatten_batch_x[:, -self.label_len:, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        if task_cond is None:
            task_cond=torch.randint(3,(1,)).to(self.device)

        y_0_hat_batch, flattern_node_enc_out = self.cond_pred_model(flatten_batch_x, dec_inp)
        y_0_hat_batch=y_0_hat_batch#[N*M,pred_len,dataset_nf] pred_len=1,
        traj_node_enc_out=flattern_node_enc_out.reshape(N,M,self.hidden_dim)

        node_enc_out=self.traj_reduc(traj_node_enc_out)#[N,hidden_dim]

        assert y_0_hat_batch.shape == flatten_batch_y.shape

        pred_y_loss = (y_0_hat_batch - flatten_batch_y).square().mean()  # mse loss
        pred_y_loss_weight = 1.0 / (
                    2 * torch.exp(self.pred_y_loss_weight)) * pred_y_loss + 0.5 * self.pred_y_loss_weight

        recon_loss = self.reconstruct_loss(node_enc_out, edge_index,
                                     batch=batch_index,
                                     )
        recon_loss_weight = 1.0 / torch.exp(self.recon_loss_weight) * recon_loss + self.recon_loss_weight
        loss_list=[pred_y_loss,recon_loss,pred_y_loss_weight + recon_loss_weight]
        return loss_list,task_cond.cpu().detach().item()
    def training_step_onegraph(self,gdata,task_cond=None):
        edge_index=gdata.edge_index.to(self.device)
        assert pyg_utils.is_undirected(edge_index),"currently only supports undirected graphs!!"
        N,M,W,F=gdata.x.shape
        assert F==self.dataset_nf
        assert W==self.windows
        o_batch_x=gdata.x #[N,M,window_size,dataset_nf]#M为观测轨迹个数

        flatten_batch_x=o_batch_x.reshape(N*M,self.windows,self.dataset_nf)
        batch_y=gdata.y #[N,M,window_size,dataset_nf]
        flatten_batch_y=batch_y.reshape(N*M,-1,self.dataset_nf)#[N*M,pred_len,dataset_nf]
        assert flatten_batch_x.shape[1] == self.windows
        assert flatten_batch_y.shape[1] == self.windows
        dec_inp_pred = torch.zeros(
            [flatten_batch_x.size(0), self.pred_len, self.dataset_nf]
        ).to(self.device)
        dec_inp_label = flatten_batch_x[:, -self.label_len:, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        if task_cond is None:
            task_cond=torch.randint(3,(1,)).to(self.device)

        y_0_hat_batch, flattern_node_enc_out = self.cond_pred_model(flatten_batch_x, dec_inp)
        y_0_hat_batch=y_0_hat_batch#[N*M,pred_len,dataset_nf] pred_len=1,
        traj_node_enc_out=flattern_node_enc_out.reshape(N,M,self.hidden_dim)

        node_enc_out=self.traj_reduc(traj_node_enc_out)#[N,hidden_dim]

        assert y_0_hat_batch.shape == flatten_batch_y.shape

        pred_y_loss = (y_0_hat_batch - flatten_batch_y).square().mean()
      #  print("pred_y_loss",pred_y_loss)# mse loss
        pred_y_loss_weight = 1.0 / (
                    2 * torch.exp(self.pred_y_loss_weight)) * pred_y_loss + 0.5 * self.pred_y_loss_weight

        recon_loss = self._compute_single_graph(node_enc_out, edge_index,
                                     )
      #  print("recon_loss",recon_loss)
        recon_loss_weight = 1.0 / torch.exp(self.recon_loss_weight) * recon_loss + self.recon_loss_weight
        loss_list=[pred_y_loss,recon_loss,pred_y_loss_weight + recon_loss_weight]
        return loss_list,task_cond.cpu().detach().item()

    def eval_pred_error(self,gdata):
        if isinstance(gdata,Batch):
            batch_index=gdata.batch.to(self.device)
        else:
            batch_index = None
        edge_index = gdata.edge_index.to(self.device)
        assert pyg_utils.is_undirected(edge_index), "currently only supports undirected graphs!!"
        N, M, W, F = gdata.x.shape
        assert F == self.dataset_nf
        assert W == self.windows
        o_batch_x = gdata.x  # [N,M,window_size,dataset_nf]#M为观测轨迹个数
        flatten_batch_x = o_batch_x.reshape(N * M, self.windows, self.dataset_nf)
        batch_y = gdata.y  # [N,M,window_size,dataset_nf]
        flatten_batch_y = batch_y.reshape(N * M, -1, self.dataset_nf)  # [N*M,pred_len,dataset_nf]
        assert flatten_batch_x.shape[1] == self.windows
        assert flatten_batch_y.shape[1] == self.windows
        dec_inp_pred = torch.zeros(
            [flatten_batch_x.size(0), self.pred_len, self.dataset_nf]
        ).to(self.device)
        dec_inp_label = flatten_batch_x[:, -self.label_len:, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        with torch.no_grad():
            y_0_hat_batch, flattern_node_enc_out = self.cond_pred_model(flatten_batch_x, dec_inp)
            y_0_hat_batch = y_0_hat_batch  # [N*M,pred_len,dataset_nf] pred_len=1,
            traj_node_enc_out = flattern_node_enc_out.reshape(N, M, self.hidden_dim)

            node_enc_out = self.traj_reduc(traj_node_enc_out)  # [N,hidden_dim]

            assert y_0_hat_batch.shape == flatten_batch_y.shape
            pred_y_loss = (y_0_hat_batch - flatten_batch_y).square().mean()  # mse loss

            if batch_index is not None:
                recon_loss = self.reconstruct_loss(node_enc_out, edge_index,
                                                                batch=batch_index,
                                                                )
            else:
                recon_loss=self._compute_single_graph(node_enc_out,edge_index,sigmoid=True)
        return pred_y_loss,recon_loss
    def eval_Zlatent(self,gdata):
        if isinstance(gdata,Batch):
            batch_index=gdata.batch.to(self.device)
        else:
            batch_index = None
        edge_index = gdata.edge_index.to(self.device)
        assert pyg_utils.is_undirected(edge_index), "currently only supports undirected graphs!!"
        N, M, W, F = gdata.x.shape
        assert F == self.dataset_nf
        assert W == self.windows
        o_batch_x = gdata.x  # [N,M,window_size,dataset_nf]#M为观测轨迹个数
        flatten_batch_x = o_batch_x.reshape(N * M, self.windows, self.dataset_nf)
        batch_y = gdata.y  # [N,M,window_size,dataset_nf]
        flatten_batch_y = batch_y.reshape(N * M, -1, self.dataset_nf)  # [N*M,pred_len,dataset_nf]
        assert flatten_batch_x.shape[1] == self.windows
        assert flatten_batch_y.shape[1] == self.windows
        dec_inp_pred = torch.zeros(
            [flatten_batch_x.size(0), self.pred_len, self.dataset_nf]
        ).to(self.device)
        dec_inp_label = flatten_batch_x[:, -self.label_len:, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        with torch.no_grad():
            flattern_node_enc_out = self.cond_pred_model.Z_latent(flatten_batch_x, dec_inp)
            traj_node_enc_out = flattern_node_enc_out.reshape(N, M, self.hidden_dim)
            Z_mean=traj_node_enc_out.mean()
            Z_var=traj_node_enc_out.var()
            node_enc_out = self.traj_reduc(traj_node_enc_out)  # [N,hidden_dim]
        return Z_mean,Z_var,node_enc_out
    def get_zlatent(self,gdata):
        if isinstance(gdata, Batch):
            batch_index = gdata.batch.to(self.device)
        else:
            batch_index = None
        edge_index = gdata.edge_index.to(self.device)
        assert pyg_utils.is_undirected(edge_index), "currently only supports undirected graphs!!"
        N, M, W, F = gdata.x.shape
        assert F == self.dataset_nf
        assert W == self.windows
        o_batch_x = gdata.x  # [N,M,window_size,dataset_nf]#M为观测轨迹个数
        flatten_batch_x = o_batch_x.reshape(N * M, self.windows, self.dataset_nf)

        assert flatten_batch_x.shape[1] == self.windows

        dec_inp_pred = torch.zeros(
            [flatten_batch_x.size(0), self.pred_len, self.dataset_nf]
        ).to(self.device)
        dec_inp_label = flatten_batch_x[:, -self.label_len:, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

        flattern_node_enc_out = self.cond_pred_model.Z_latent(flatten_batch_x, dec_inp)

        traj_node_enc_out = flattern_node_enc_out.reshape(N, M, self.hidden_dim)

        node_enc_out = self.traj_reduc(traj_node_enc_out)  # [N,hidden_dim]
        return node_enc_out

    def reconstruct_loss(self, node_embeddings,edge_index, batch=None,reduction='mean'):

        batch_indices = batch

        # 分组计算
        pyg_data_list = []
        losses = []
        unbatch_edge_index = pyg_utils.unbatch_edge_index(edge_index,batch_indices)

        for b, edge_index in enumerate(unbatch_edge_index):
            mask = (batch_indices == b)
            emb = node_embeddings[mask]

            # 计算上三角邻接矩阵
            recon_loss = self._compute_single_graph(emb,edge_index,sigmoid=True)

            losses.append(recon_loss)
        loss_tensor = torch.stack(losses) if losses else torch.tensor(0.0)
        # 根据reduction参数缩减损失
        if reduction == 'mean':
            total_recon_loss = loss_tensor.mean()
        elif reduction == 'sum':
            total_recon_loss = loss_tensor.sum()
        else:  # 'none'
            total_recon_loss = loss_tensor

        return  total_recon_loss

    def _compute_single_graph(self, embeddings,edge_index,sigmoid=True,train=True):
        """计算单个图的上三角邻接矩阵"""
        n_nodes = embeddings.shape[0]
        n_edges=edge_index.shape[1]
       # print("n node:{}".format(n_nodes))

        if n_nodes <= 1:
            return torch.zeros(0, 0, device=embeddings.device)


        if pyg_utils.is_undirected(edge_index):
            pos_mask=edge_index[0]<edge_index[1]#无向图,仅保留上三角的正样本
            pos_edge_index=edge_index[:,pos_mask]
            num_neg_samples=round(self.neg_samples_ratio*edge_index.size(1))
            neg_edge_index = negative_sampling(pos_edge_index,
                                               num_neg_samples=num_neg_samples,
                                               force_undirected=True
                                               )
            neg_mask = neg_edge_index[0] < neg_edge_index[1]  # 无向图,仅保留上三角的负样本
            neg_edge_index = neg_edge_index[:, neg_mask]
        else:
            raise ValueError('currently only supports undirected graphs!!')


        # 计算上三角索引
        rows, cols = torch.triu_indices(n_nodes, n_nodes, offset=1)

        full_edge_index=torch.stack([rows,cols], dim=0)


        similarities = self.edge_decoder(embeddings, full_edge_index, sigmoid=sigmoid)



        # 创建稀疏上三角矩阵
        adj_triu_prob = torch.zeros(n_nodes, n_nodes, device=embeddings.device)
        adj_triu_prob[rows, cols] = similarities
        # 对称化（如果是无向图）
        adj_prob = torch.max(adj_triu_prob, adj_triu_prob.t())
        pos_logits=adj_prob[pos_edge_index[0],pos_edge_index[1]]+EPS
        pos_loss = -torch.log(
            pos_logits).mean()
        neg_logits=adj_prob[neg_edge_index[0],neg_edge_index[1]]+EPS
        neg_loss = -torch.log(1 - neg_logits).mean()
        recon_loss = pos_loss + neg_loss

        # 计算预测度：每个节点的边概率和 [N]
        if train and self.degree_loss_weight>0:
            d_pred = adj_prob.sum(dim=1)
           # print("d_pred",d_pred.shape)
            true_degrees = pyg_utils.degree(edge_index[0],num_nodes=n_nodes, dtype=d_pred.dtype)
           # print("true_degrees",true_degrees.shape)
            degree_loss = F.mse_loss(d_pred, true_degrees)/n_edges
            recon_loss =(recon_loss + self.degree_loss_weight * degree_loss)/(1+self.degree_loss_weight)


        return recon_loss
    def _sample_single_graph(self, embeddings):
        """计算单个图的上三角邻接矩阵"""
        n_nodes = embeddings.shape[0]



        # 计算上三角索引
        rows, cols = torch.triu_indices(n_nodes, n_nodes, offset=1)

        full_edge_index=torch.stack([rows,cols], dim=0).to(embeddings.device)

        similarities = self.edge_decoder(embeddings, full_edge_index, sigmoid=True)
        edge_samples = torch.bernoulli(similarities)
        edge_mask = edge_samples > 0.5
        # 获取存在的边索引
        edge_index = full_edge_index[:, edge_mask]
        # 构建无向图
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        return Data(x=embeddings, edge_index=edge_index)




