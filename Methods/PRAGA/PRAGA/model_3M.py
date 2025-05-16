import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class Encoder_overall(Module):
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dim_in_feat_omics2, dim_out_feat_omics2,
                 dim_in_feat_omics3, dim_out_feat_omics3, dropout=0.0, act=F.relu):
        super(Encoder_overall, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_in_feat_omics3 = dim_in_feat_omics3
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.dim_out_feat_omics3 = dim_out_feat_omics3
        self.dropout = dropout
        self.act = act

        self.conv1X1_omics1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv1X1_omics2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv1X1_omics3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.encoder_omics1 = Encoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
        self.decoder_omics1 = Decoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1)
        self.encoder_omics2 = Encoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2)
        self.decoder_omics2 = Decoder(self.dim_out_feat_omics2, self.dim_in_feat_omics2)
        self.encoder_omics3 = Encoder(self.dim_in_feat_omics3, self.dim_out_feat_omics3)
        self.decoder_omics3 = Decoder(self.dim_out_feat_omics3, self.dim_in_feat_omics3)

        self.MLP = MLP(self.dim_out_feat_omics1 * 3, self.dim_out_feat_omics1, self.dim_out_feat_omics1)

    def forward(self, features_omics1, features_omics2, features_omics3, adj_spatial_omics1, adj_feature_omics1,
                adj_spatial_omics2, adj_feature_omics2, adj_spatial_omics3, adj_feature_omics3):

        _adj_spatial_omics1 = adj_spatial_omics1.unsqueeze(0)  # shape: (1, N, N)
        _adj_feature_omics1 = adj_feature_omics1.unsqueeze(0)  # shape: (1, N, N)

        _adj_spatial_omics2 = adj_spatial_omics2.unsqueeze(0)  # shape: (1, N, N)
        _adj_feature_omics2 = adj_feature_omics2.unsqueeze(0)  # shape: (1, N, N)

        _adj_spatial_omics3 = adj_spatial_omics3.unsqueeze(0)  # shape: (1, N, N)
        _adj_feature_omics3 = adj_feature_omics3.unsqueeze(0)  # shape: (1, N, N)

        # shape: (2, N, N)
        cat_adj_omics1 = torch.cat((_adj_spatial_omics1, _adj_feature_omics1), dim=0)
        cat_adj_omics2 = torch.cat((_adj_spatial_omics2, _adj_feature_omics2), dim=0)
        cat_adj_omics3 = torch.cat((_adj_spatial_omics3, _adj_feature_omics3), dim=0)

        adj_feature_omics1 = self.conv1X1_omics1(cat_adj_omics1).squeeze(0)
        adj_feature_omics2 = self.conv1X1_omics2(cat_adj_omics2).squeeze(0)
        adj_feature_omics3 = self.conv1X1_omics3(cat_adj_omics3).squeeze(0)

        # graph2
        emb_latent_omics1 = self.encoder_omics1(features_omics1, adj_feature_omics1)
        emb_latent_omics2 = self.encoder_omics2(features_omics2, adj_feature_omics2)
        emb_latent_omics3 = self.encoder_omics3(features_omics3, adj_feature_omics3)

        cat_emb_latent = torch.cat((emb_latent_omics1, emb_latent_omics2, emb_latent_omics3), dim=1)

        emb_latent_combined = self.MLP(cat_emb_latent)

        # reverse the integrated representation back into the original expression space with modality-specific decoder
        emb_recon_omics1 = self.decoder_omics1(emb_latent_combined, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(emb_latent_combined, adj_spatial_omics2)
        emb_recon_omics3 = self.decoder_omics3(emb_latent_combined, adj_spatial_omics3)

        results = {'emb_latent_omics1': emb_latent_omics1,
                   'emb_latent_omics2': emb_latent_omics2,
                   'emb_latent_omics3': emb_latent_omics3,
                   'emb_latent_combined': emb_latent_combined,
                   'emb_recon_omics1': emb_recon_omics1,
                   'emb_recon_omics2': emb_recon_omics2,
                   'emb_recon_omics3': emb_recon_omics3,
                   }

        return results

'''
---------------------
Encoder & Decoder functions
author: Yahui Long https://github.com/JinmiaoChenLab/SpatialGlue
AGPL-3.0 LICENSE
---------------------
'''

class Encoder(Module):
    """\
    Modality-specific GNN encoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.

    Returns
    -------
    Latent representation.

    """
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        feat_embeding = torch.mm(feat, self.weight)
        feat_embeding = torch.spmm(adj, feat_embeding)
        return feat_embeding


class Decoder(Module):
    """\
    Modality-specific GNN decoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.

    Returns
    -------
    Reconstructed representation.

    """
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        return x




class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out





