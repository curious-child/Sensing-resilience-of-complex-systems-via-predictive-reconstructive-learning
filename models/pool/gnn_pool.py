import torch_geometric.nn as gnn
import torch
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F
def gnn_pool(pool_name):
    if pool_name=="add":
        return gnn.global_add_pool
    elif pool_name=="max":
        return gnn.global_max_pool
    elif pool_name=="mean":
        return gnn.global_mean_pool
    else:
        raise ValueError("the definition  don't exit\n"
                         "\tyou can define it before using it")
class SelfAttentionPooling(nn.Module):
    def __init__(self, emd_dim):
        super().__init__()
        # 一个简单的线性层来生成注意力分数
        self.attention = nn.Linear(emd_dim, 1)

    def forward(self, x):
        # x shape: [batch, time_dim, emd_dim]
        # 计算每个时间步的注意力分数 [batch, time_dim, 1]
       # assert x.dim() == 3
        attn_scores = self.attention(x)
        # 使用softmax将分数转换为权重，和为1 [batch, time_dim, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        # 进行加权求和 [batch, 1, emd_dim]
        pooled = torch.bmm(attn_weights.transpose(1, 2), x)
        # 压缩维度 [batch, emd_dim]
        pooled = pooled.squeeze(1)
        return pooled  # 返回聚合结果和注意力权重（可用于分析）