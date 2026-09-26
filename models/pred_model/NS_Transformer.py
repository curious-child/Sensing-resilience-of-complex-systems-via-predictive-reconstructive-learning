import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_timeseries.nn.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from torch_timeseries.nn.SelfAttention_Family import DSAttention, AttentionLayer
from torch_timeseries.nn.embedding import DataEmbedding

from models.layer.gnn_conv import gnn_conv
from models.pool.gnn_pool import SelfAttentionPooling
class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y

class SpatialBlock(nn.Module):
    def __init__(self, c_in, c_out,gnn_name, gnn_param ):
        super(SpatialBlock, self).__init__()
        self.gnn=gnn_conv(gnn_name=gnn_name,in_channels=c_in,out_channels=c_out,gnn_param=gnn_param,)

    def forward(self, x, edge_index):
        # x: [B*V,c_in]
        # edge_index: [2,E]

        return torch.relu(self.gnn(x,edge_index))# [B*V,c_out]
class ns_Transformer(nn.Module):
    """
    Non-stationary Transformer
    """

    def __init__(self, configs):
        super(ns_Transformer, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len



        # Embedding
        self.enc_embedding = DataEmbedding(configs.dataset_nf, configs.d_model,
                                           configs.dropout,time_embed=False)#without time_mark
        self.dec_embedding = DataEmbedding(configs.dataset_nf, configs.d_model,
                                           configs.dropout,time_embed=False)#without time_mark
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.state_decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.dataset_nf, bias=True)
        )

        self.tau_learner = Projector(enc_in=configs.dataset_nf, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims,
                                     hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.dataset_nf, seq_len=configs.seq_len,
                                       hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                       output_dim=configs.seq_len)
        self.node_enc_reduc=SelfAttentionPooling(
                                         emd_dim=configs.d_model,
                                         )



        # self.z_mean = nn.Sequential(
        #     nn.Linear(configs.d_model, configs.d_model),
        #     nn.ReLU(),
        #     nn.Linear(configs.d_model, configs.d_model)
        # )
        # self.z_logvar = nn.Sequential(
        #     nn.Linear(configs.d_model, configs.d_model),
        #     nn.ReLU(),
        #     nn.Linear(configs.d_model, configs.d_model)
        # )
        #
        # self.z_out = nn.Sequential(
        #     nn.Linear(configs.d_model, configs.d_model),
        #     nn.ReLU(),
        #     nn.Linear(configs.d_model, configs.d_model)
        # )

    def KL_loss_normal(self, posterior_mean, posterior_logvar):
        KL = -0.5 * torch.mean(1 - posterior_mean ** 2 + posterior_logvar -
                               torch.exp(posterior_logvar), dim=1)
        return torch.mean(KL)
    


    def reparameterize(self, posterior_mean, posterior_logvar):
        posterior_var = posterior_logvar.exp()
        # take sample
        if self.training:
            posterior_mean = posterior_mean.repeat(100, 1, 1, 1)
            posterior_var = posterior_var.repeat(100, 1, 1, 1)
            eps = torch.zeros_like(posterior_var).normal_()
            z = posterior_mean + posterior_var.sqrt() * eps  # reparameterization
            z = z.mean(0)
        else:
            z = posterior_mean
        # z = posterior_mean
        return z

    def forward(self, x_enc,  x_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
       # print(x_enc.type(), x_dec.type())
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc
        x_dec_new = torch.cat([x_enc[:, -self.label_len:, :], torch.zeros_like(x_dec[:, -self.pred_len:, :])],
                              dim=1).to(x_enc.device).clone()

        tau = self.tau_learner(x_raw, std_enc).exp()  # B x S x E, B x 1 x E -> B x 1, positive scalar
        delta = self.delta_learner(x_raw, mean_enc)  # B x S x E, B x 1 x E -> B x S

        # Model Inference
        enc_out = self.enc_embedding(x=x_enc, x_mark=None)
        #enc_out [B,S,d_model]??
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)
        #print("enc_out.shape",enc_out.shape) #?
        node_enc_state = self.node_enc_reduc(enc_out) # [B, d_model]




        dec_out = self.dec_embedding(x=x_dec_new, x_mark=None)
        dec_out = self.state_decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau, delta=delta)
        # De-normalization
        dec_out = dec_out * std_enc + mean_enc
        node_pred_state = dec_out[:, -self.pred_len:, :]  # [B, L, D]


        return node_pred_state, node_enc_state  # [B, L, D]

    def Z_latent(self,x_enc, x_dec,
                 enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # print(x_enc.type(), x_dec.type())
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc
        x_dec_new = torch.cat([x_enc[:, -self.label_len:, :], torch.zeros_like(x_dec[:, -self.pred_len:, :])],
                              dim=1).to(x_enc.device).clone()

        tau = self.tau_learner(x_raw, std_enc).exp()  # B x S x E, B x 1 x E -> B x 1, positive scalar
        delta = self.delta_learner(x_raw, mean_enc)  # B x S x E, B x 1 x E -> B x S

        # Model Inference
        enc_out = self.enc_embedding(x=x_enc, x_mark=None)
        # enc_out [B,S,d_model]??
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)
        # print("enc_out.shape",enc_out.shape) #?
        node_enc_state = self.node_enc_reduc(enc_out)  # [B, d_model]
        return node_enc_state
class ns_Transformer_VAE(nn.Module):
    """
    Non-stationary Transformer
    """

    def __init__(self, configs):
        super(ns_Transformer_VAE, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len

        # Embedding
        self.enc_embedding = DataEmbedding(configs.dataset_nf, configs.d_model,
                                           configs.dropout, time_embed=False)  # without time_mark
        self.dec_embedding = DataEmbedding(configs.dataset_nf, configs.d_model,
                                           configs.dropout, time_embed=False)  # without time_mark
        self.cond_embedding = nn.Embedding(3, configs.d_model)
        self.condition_encoder = nn.Sequential(
            nn.Linear(2*configs.d_model, configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model, configs.d_model),
            nn.LayerNorm(configs.d_model)
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.encoder_mean = nn.Linear(configs.d_model, configs.d_model)
        self.encoder_logvar = nn.Linear(configs.d_model, configs.d_model)
        # Decoder
        self.state_decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.dataset_nf, bias=True)
        )

        self.tau_learner = Projector(enc_in=configs.dataset_nf, seq_len=configs.seq_len,
                                     hidden_dims=configs.p_hidden_dims,
                                     hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.dataset_nf, seq_len=configs.seq_len,
                                       hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                       output_dim=configs.seq_len)
        self.node_enc_reduc = SelfAttentionPooling(
            emd_dim=configs.d_model,
        )

        # self.z_mean = nn.Sequential(
        #     nn.Linear(configs.d_model, configs.d_model),
        #     nn.ReLU(),
        #     nn.Linear(configs.d_model, configs.d_model)
        # )
        # self.z_logvar = nn.Sequential(
        #     nn.Linear(configs.d_model, configs.d_model),
        #     nn.ReLU(),
        #     nn.Linear(configs.d_model, configs.d_model)
        # )
        #
        # self.z_out = nn.Sequential(
        #     nn.Linear(configs.d_model, configs.d_model),
        #     nn.ReLU(),
        #     nn.Linear(configs.d_model, configs.d_model)
        # )

    def KL_loss_normal(self, posterior_mean, posterior_logvar):
        KL = -0.5 * torch.mean(1 - posterior_mean ** 2 + posterior_logvar -
                               torch.exp(posterior_logvar), dim=1)
        return torch.mean(KL)

    def reparameterize(self, posterior_mean, posterior_logvar):
        posterior_var = posterior_logvar.exp()
        # take sample
        if self.training:
            posterior_mean = posterior_mean.repeat(10, 1, 1, 1)
            posterior_var = posterior_var.repeat(10, 1, 1, 1)
            eps = torch.zeros_like(posterior_var).normal_()
            z = posterior_mean + posterior_var.sqrt() * eps  # reparameterization
            z = z.mean(0)
        else:
            z = posterior_mean
        # z = posterior_mean
        return z


    def forward(self, x_enc, x_dec,cond,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # print(x_enc.type(), x_dec.type())
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc
        x_dec_new = torch.cat([x_enc[:, -self.label_len:, :], torch.zeros_like(x_dec[:, -self.pred_len:, :])],
                              dim=1).to(x_enc.device).clone()

        tau = self.tau_learner(x_raw, std_enc).exp()  # B x S x E, B x 1 x E -> B x 1, positive scalar
        delta = self.delta_learner(x_raw, mean_enc)  # B x S x E, B x 1 x E -> B x S

        # Model Inference
        enc_out = self.enc_embedding(x=x_enc, x_mark=None)
        cond_emb=self.cond_embedding(cond)#[1,d_model]
       # print("cond shape {}".format(cond_emb.shape))
        cond_enc_out=torch.cat([enc_out, cond_emb.unsqueeze(0).expand(enc_out.shape[0], enc_out.shape[1], -1)], dim=2)

        enc_out=self.condition_encoder(cond_enc_out)
        # enc_out [B,S,d_model]??
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)
        enc_out_mean=self.encoder_mean(enc_out)
        enc_out_var=self.encoder_logvar(enc_out)

        enc_out=self.reparameterize(enc_out_mean,enc_out_var)
        # print("enc_out.shape",enc_out.shape) #?
        node_enc_state = self.node_enc_reduc(enc_out)  # [B, d_model]


        dec_out = self.dec_embedding(x=x_dec_new, x_mark=None)
        dec_out = self.state_decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau,
                                     delta=delta)
        # De-normalization
        dec_out = dec_out * std_enc + mean_enc
        node_pred_state = dec_out[:, -self.pred_len:, :]  # [B, L, D]

        return node_pred_state, node_enc_state  # [B, L, D]
    def Z_latent(self,x_enc, x_dec,cond,
                 enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # print(x_enc.type(), x_dec.type())
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc
        x_dec_new = torch.cat([x_enc[:, -self.label_len:, :], torch.zeros_like(x_dec[:, -self.pred_len:, :])],
                              dim=1).to(x_enc.device).clone()

        tau = self.tau_learner(x_raw, std_enc).exp()  # B x S x E, B x 1 x E -> B x 1, positive scalar
        delta = self.delta_learner(x_raw, mean_enc)  # B x S x E, B x 1 x E -> B x S

        # Model Inference
        enc_out = self.enc_embedding(x=x_enc, x_mark=None)
        cond_emb = self.cond_embedding(cond)  # [1,d_model]
        # print("cond shape {}".format(cond_emb.shape))
        cond_enc_out = torch.cat([enc_out, cond_emb.unsqueeze(0).expand(enc_out.shape[0], enc_out.shape[1], -1)], dim=2)

        enc_out = self.condition_encoder(cond_enc_out)
        # enc_out [B,S,d_model]??
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)

        enc_out_mean = self.encoder_mean(enc_out)
        enc_out_var = self.encoder_logvar(enc_out)
        node_enc_mean = self.node_enc_reduc(enc_out_mean)  # [B, d_model]
        node_enc_var = self.node_enc_reduc(enc_out_var)
        return node_enc_mean,node_enc_var
