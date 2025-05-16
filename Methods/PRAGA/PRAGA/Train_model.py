import torch
import time
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from .model import Encoder_overall
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from .preprocess import adjacent_matrix_preprocessing
from .optimal_clustering_HLN import R5

class Train:
    def __init__(self, 
        data,
        datatype,
        device,
        random_seed = 2024,
        dim_input=3000,
        dim_output=64,
        Arg=None
        ):

        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.dim_input = dim_input
        self.dim_output = dim_output
        
        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to_dense().to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to_dense().to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to_dense().to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to_dense().to(self.device)

        self.paramed_adj_omics1 = Parametered_Graph(self.adj_feature_omics1, self.device).to(self.device)
        self.paramed_adj_omics2 = Parametered_Graph(self.adj_feature_omics2, self.device).to(self.device)

        self.adj_feature_omics1_copy = copy.deepcopy(self.adj_feature_omics1)
        self.adj_feature_omics2_copy = copy.deepcopy(self.adj_feature_omics2)

        self.EMA_coeffi = 0.9
        self.K = 5
        self.T = 4
        self.arg = Arg

        self.clustering = R5(self.datatype, self.arg)
        
        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)
        
        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs
        
        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output
        
        if self.datatype == 'SPOTS':
           self.epochs = 200
           self.weight_factors = [Arg.RNA_weight, Arg.ADT_weight]
           self.weight_decay = 5e-3
           self.learning_rate = 0.01
           
        elif self.datatype == 'Stereo-CITE-seq':
           self.epochs = 300
           self.weight_factors = [Arg.RNA_weight, Arg.ADT_weight]
           self.weight_decay = 5e-2
           self.learning_rate = 0.01
           
        elif self.datatype == '10x':
           self.learning_rate = 0.01
           self.epochs = 30
           self.weight_factors = [Arg.RNA_weight, Arg.ADT_weight]
           self.weight_decay = 5e-3
           self.EMA_coeffi = Arg.alpha
            
        elif self.datatype == 'Spatial-epigenome-transcriptome': 
           self.epochs = 300
           self.weight_factors = [Arg.RNA_weight, Arg.ADT_weight]
           self.learning_rate = 0.01
           self.weight_decay = 5e-2

        elif self.datatype == 'Sim_L_ADT' or self.datatype == 'Sim_L_ATAC':
            self.learning_rate = 0.0001
            self.epochs = 50
            self.weight_factors = [5, 1]
            self.weight_decay = 0
            self.EMA_coeffi = Arg.alpha
        elif self.datatype == 'HT':
            self.learning_rate = 0.001
            self.epochs = 30
            self.weight_factors = [Arg.RNA_weight, Arg.ADT_weight]
            self.weight_decay = 5e-3
            self.EMA_coeffi = Arg.alpha
        elif self.datatype == 'MISAR':
           self.learning_rate = 0.0001
           self.epochs = 200
           self.weight_factors = [1, 1]
           self.weight_decay = 0
           self.EMA_coeffi = Arg.alpha
    
    def train(self):

        self.model = Encoder_overall(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2).to(self.device)
        self.optimizer = torch.optim.SGD(list(self.model.parameters()) +
                                          list(self.paramed_adj_omics1.parameters()) +
                                          list(self.paramed_adj_omics2.parameters()),
                                          lr=self.learning_rate,
                                          momentum=0.9,
                                          weight_decay=self.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        self.model.train()
        from torch.cuda.amp import GradScaler, autocast
        train_time_start = time.time()
        for epoch in tqdm(range(self.epochs)):
            scaler = torch.cuda.amp.GradScaler()
            with autocast():
                self.model.train()

                results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2)

                # reconstruction loss
                self.loss_recon_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_omics1'])
                self.loss_recon_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_omics2'])

                loss = self.weight_factors[0]*self.loss_recon_omics1 + self.weight_factors[1]*self.loss_recon_omics2

                updated_adj_omics1 = self.paramed_adj_omics1()
                updated_adj_omics2 = self.paramed_adj_omics2()

                loss_fro = (torch.norm(updated_adj_omics1 - self.adj_feature_omics1_copy.detach(), p='fro') +
                            torch.norm(updated_adj_omics2 - self.adj_feature_omics2_copy.detach(), p='fro'))/2

                clustering_loss = self.clustering(results['emb_latent_combined'], epoch)

                print(loss, loss_fro, clustering_loss)

                loss = loss + clustering_loss + loss_fro

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # scheduler.step()


            self.adj_feature_omics1 = self.paramed_adj_omics1()
            self.adj_feature_omics2 = self.paramed_adj_omics2()

            self.adj_feature_omics1_copy = self.EMA_coeffi * self.adj_feature_omics1_copy + (
                    1 - self.EMA_coeffi) * self.adj_feature_omics1.detach().clone()
            self.adj_feature_omics2_copy = self.EMA_coeffi * self.adj_feature_omics2_copy + (
                    1 - self.EMA_coeffi) * self.adj_feature_omics2.detach().clone()

        print("Model training finished!\n")
        train_time_end = time.time()

        print("Training time: {}".format(train_time_end-train_time_start))

        start_time = time.time()
    
        with torch.no_grad():
          self.model.eval()
          results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2)

        end_time = time.time()
        print("Infer time: ", end_time - start_time)

        emb_omics1 = F.normalize(results['emb_latent_omics1'].squeeze(0), p=2, eps=1e-12, dim=1)
        emb_omics2 = F.normalize(results['emb_latent_omics2'].squeeze(0), p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'].squeeze(0), p=2, eps=1e-12, dim=1)

        A_no_diag = self.paramed_adj_omics2().cpu().detach().clone()
        A_no_diag.fill_diagonal_(0)
        
        output = {'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
                  'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
                  'recon_omics1': results['emb_recon_omics1'].detach().cpu().numpy(),
                  'recon_omics2': results['emb_recon_omics2'].detach().cpu().numpy(),
                  'PRAGA': emb_combined.detach().cpu().numpy(),
                  'adj_feature_omics1': self.adj_feature_omics1.detach().cpu().numpy()
        }
        
        return output
    
class Parametered_Graph(nn.Module):
    def __init__(self, adj, device):
        super(Parametered_Graph, self).__init__()
        self.adj = adj
        self.device = device

        n = self.adj.shape[0]
        self.paramed_adj_omics = nn.Parameter(torch.FloatTensor(n, n))
        self.paramed_adj_omics.data.copy_(self.adj)

    def forward(self, A=None):
        if A is None:
            adj = (self.paramed_adj_omics + self.paramed_adj_omics.t()) / 2
        else:
            adj = (A + A.t()) / 2

        adj = nn.ReLU(inplace=True)(adj)
        normalized_adj = self._normalize(adj.to(self.device) + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj.to(self.device)

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx



    
    
      

    
        
    
    
