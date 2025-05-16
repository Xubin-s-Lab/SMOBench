import os
import torch
import pandas as pd
import scanpy as sc
import SpatialGlue
import argparse
import time
# from SpatialGlue.preprocess import clr_normalize_each_cell, pca, lsi
# Environment configuration. SpatialGlue pacakge can be implemented with either CPU or GPU. GPU acceleration is highly recommend for imporoved efficiency.


# the location of R, which is required for the 'mclust' algorithm. Please replace the path below with local R installation path
# os.environ['R_HOME'] = '/scbio4/tools/R/R-4.0.3_openblas/R-4.0.3'

def main(args):
    # read data
    adata_omics1 = sc.read_h5ad(args.RNA_path)
    if args.ADT_path != '':
        adata_omics2 = sc.read_h5ad(args.ADT_path)
        modality = 'ADT'
    elif args.ATAC_path != '':
        adata_omics2 = sc.read_h5ad(args.ATAC_path)
        modality = 'ATAC'

    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()

    # Specify data type
    data_type = '10x'

    # Fix random seed
    from SpatialGlue.preprocess import fix_seed
    fix_seed(args.seed)

    from SpatialGlue.preprocess import clr_normalize_each_cell, pca, lsi


    # RNA
    sc.pp.filter_genes(adata_omics1, min_cells=10)
    sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.scale(adata_omics1)

    if args.data_type=='SPOTS':
        n_comps = 20
    elif args.data_type=='Stereo-CITE-seq':
        n_comps = 10
    else:
        n_comps = 30

    adata_omics1_high =  adata_omics1[:, adata_omics1.var['highly_variable']]
    adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=n_comps)

    if modality == 'ADT':
        # Protein
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=n_comps)
    elif modality == 'ATAC':
        # adata_omics2 = adata_omics2[
        #     adata_omics1.obs_names].copy()  # .obsm['X_lsi'] represents the dimension reduced feature
        if 'X_lsi' not in adata_omics2.obsm.keys():
            sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
            lsi(adata_omics2, use_highly_variable=False, n_components=n_comps)

        adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()

    from SpatialGlue.preprocess import construct_neighbor_graph
    data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type)

    # define model
    from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    model = Train_SpatialGlue(data, datatype=data_type, device=device)

    # train model
    start_time = time.time()
    output = model.train()
    end_time = time.time()
    print('training time:', end_time - start_time)

    adata = adata_omics1.copy()
    adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
    adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
    adata.obsm['emb_recon_omics1'] = output['recon_omics1'].copy()
    adata.obsm['emb_recon_omics2'] = output['recon_omics2'].copy()
    adata.obsm['SpatialGlue'] = output['SpatialGlue'].copy()

    adata.write(args.save_path)
    print(adata)
    print('Saving results to...', args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to modify global variable')
    parser.add_argument('--data_type', type=str, default='A1', help='weight')
    parser.add_argument('--RNA_path', default='', type=str, help='RNA_path')
    parser.add_argument('--ADT_path', default='', type=str, help='ADT_path')
    parser.add_argument('--ATAC_path', default='', type=str, help='ATAC_path')
    parser.add_argument('--save_path', default='', type=str, help='save_path')
    parser.add_argument('--seed', default=2024, type=int, help='seed')
    parser.add_argument('--device', default='cuda:0', type=str, help='seed')
    args = parser.parse_args()
    main(args)