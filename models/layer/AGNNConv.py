import functools
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_sparse import SparseTensor
from torch_sparse import sum as sparse_sum
from torch_sparse import mean as sparse_mean
from torch_sparse import max as sparse_max

class ScalarEmbeddingSine1D(nn.Module):
  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    x_embed = x
    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

    pos_x = x_embed[:, None] / dim_t
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
    return pos_x

class AGNNConv(nn.Module):
  """Configurable GNN Layer
  Implements the Gated Graph ConvNet layer:
      h_i = ReLU ( U*h_i + Aggr.( sigma_ij, V*h_j) ),
      sigma_ij = sigmoid( A*h_i + B*h_j + C*e_ij ),
      e_ij = ReLU ( A*h_i + B*h_j + C*e_ij ),
      where Aggr. is an aggregation function: sum/mean/max.
  References:
      - X. Bresson and T. Laurent. An experimental study of neural networks for variable graphs. In International Conference on Learning Representations, 2018.
      - V. P. Dwivedi, C. K. Joshi, T. Laurent, Y. Bengio, and X. Bresson. Benchmarking graph neural networks. arXiv preprint arXiv:2003.00982, 2020.
  """

  def __init__(self, in_channels,out_channels, mode="residual",sparse=False,aggregation="sum", norm="batch", learn_norm=True, track_norm=False, gated=True):
    """
    Args:
        hidden_dim: Hidden dimension size (int)
        aggregation: Neighborhood aggregation scheme ("sum"/"mean"/"max")
        norm: Feature normalization scheme ("layer"/"batch"/None)
        learn_norm: Whether the normalizer has learnable affine parameters (True/False)
        track_norm: Whether batch statistics are used to compute normalization mean/std (True/False)
        gated: Whether to use edge gating (True/False)
    """
    super(AGNNConv, self).__init__()
    self.inchannels = in_channels
    self.out_channels = out_channels

    self.aggregation = aggregation
    self.norm = norm
    self.learn_norm = learn_norm
    self.track_norm = False
    self.gated = True
    self.mode=mode
    self.sparse=sparse
    assert self.gated, "Use gating with GCN, pass the `--gated` flag"

    self.U = nn.Linear(self.inchannels, self.out_channels, bias=True)
    self.V = nn.Linear(self.inchannels, self.out_channels, bias=True)
    self.A = nn.Linear(self.inchannels, self.out_channels, bias=True)
    self.B = nn.Linear(self.inchannels, self.out_channels, bias=True)
    self.C = nn.Linear(self.inchannels, self.out_channels, bias=True)
    self.pos_embed = ScalarEmbeddingSine1D(self.out_channels, normalize=False)

    self.norm_h = {
        "layer": nn.LayerNorm(self.out_channels, elementwise_affine=learn_norm),
        "batch": nn.BatchNorm1d(self.out_channels, affine=learn_norm, track_running_stats=track_norm)
    }.get(self.norm, None)

    self.norm_e = {
        "layer": nn.LayerNorm(self.out_channels, elementwise_affine=learn_norm),
        "batch": nn.BatchNorm1d(self.out_channels, affine=learn_norm, track_running_stats=track_norm)
    }.get(self.norm, None)

  def forward(self, h, edge_index , e):
    """
    Args
        In Sparse version:
          h: Input node features (V x H)
          e: Input edge features (E x H)
          edge_index: Edge indices (2 x E)
        sparse: Whether to use sparse tensors (True/False)
    Returns:
        Updated node and edge features
    """
    # if not h.shape[-1] ==self.out_channels:
    #   h=self.pos_embed(h.reshape(-1))

    adj_matrix = SparseTensor(
      row=edge_index[0],
      col=edge_index[1],
      value=torch.ones_like(edge_index[0].float()),
      sparse_sizes=(h.shape[0], h.shape[0]),
    )
    adj_matrix = adj_matrix.to(h.device)
    graph=adj_matrix

    if not self.sparse:
      batch_size, num_nodes, hidden_dim = h.shape
    else:
      batch_size = None
      num_nodes, hidden_dim = h.shape

    h_in = h
    e_in = e

    # Linear transformations for node update
    Uh = self.U(h)  # B x V x H

    if not self.sparse:
      Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1)  # B x V x V x H
    else:
      Vh = self.V(h[edge_index[1]])  # E x H

    # Linear transformations for edge update and gating
    Ah = self.A(h)  # B x V x H, source
    Bh = self.B(h)  # B x V x H, target
    Ce = self.C(e)  # B x V x V x H / E x H

    # Update edge features and compute edge gates
    if not self.sparse:
      e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H
    else:
      e = Ah[edge_index[1]] + Bh[edge_index[0]] + Ce  # E x H

    gates = torch.sigmoid(e)  # B x V x V x H / E x H

    # Update node features
    h = Uh + self.aggregate(Vh, graph, gates, edge_index=edge_index, sparse=self.sparse)  # B x V x H
    # Normalize node features
    if not self.sparse:
      h = self.norm_h(
          h.view(batch_size * num_nodes, hidden_dim)
      ).view(batch_size, num_nodes, hidden_dim) if self.norm_h else h
    else:
      h = self.norm_h(h) if self.norm_h else h

    # Normalize edge features
    if not self.sparse:
      e = self.norm_e(
          e.view(batch_size * num_nodes * num_nodes, hidden_dim)
      ).view(batch_size, num_nodes, num_nodes, hidden_dim) if self.norm_e else e
    else:
      e = self.norm_e(e) if self.norm_e else e

    # Apply non-linearity
    h = F.relu(h)
    e = F.relu(e)

    # Make residual connection
    if self.mode == "residual":
      h = h_in + h
      e = e_in + e

    return h,e
  def aggregate(self, Vh, graph, gates, mode=None, edge_index=None, sparse=False):
    """
    Args:
        In Dense version:
          Vh: Neighborhood features (B x V x V x H)
          graph: Graph adjacency matrices (B x V x V)
          gates: Edge gates (B x V x V x H)
          mode: str
        In Sparse version:
          Vh: Neighborhood features (E x H)
          graph: torch_sparse.SparseTensor (E edges for V x V adjacency matrix)
          gates: Edge gates (E x H)
          mode: str
          edge_index: Edge indices (2 x E)
        sparse: Whether to use sparse tensors (True/False)
    Returns:
        Aggregated neighborhood features (B x V x H)
    """
    # Perform feature-wise gating mechanism
    Vh = gates * Vh  # B x V x V x H

    # Enforce graph structure through masking
    # Vh[graph.unsqueeze(-1).expand_as(Vh)] = 0

    # Aggregate neighborhood features
    if not sparse:
      if (mode or self.aggregation) == "mean":
        return torch.sum(Vh, dim=2) / (torch.sum(graph, dim=2).unsqueeze(-1).type_as(Vh))
      elif (mode or self.aggregation) == "max":
        return torch.max(Vh, dim=2)[0]
      else:
        return torch.sum(Vh, dim=2)
    else:
      sparseVh = SparseTensor(
          row=edge_index[0],
          col=edge_index[1],
          value=Vh,
          sparse_sizes=(graph.size(0), graph.size(1))
      )

      if (mode or self.aggregation) == "mean":
        return sparse_mean(sparseVh, dim=1)

      elif (mode or self.aggregation) == "max":
        return sparse_max(sparseVh, dim=1)

      else:
        return sparse_sum(sparseVh, dim=1)