import argparse
import scanpy as sc
import numpy as np
from src.clustering import RunLeiden, RunLouvain, knn_adj_matrix
from src.demo import eval, eval_BC

def HLN():
    # emb_A1 = np.load('/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/yjxiao/spaVAE/src/spaMultiVAE/results/HLN/A1/final_latent.npy')
    # emb_D1 = np.load('/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/yjxiao/spaVAE/src/spaMultiVAE/results/HLN/D1/final_latent.npy')
    # emb = np.concatenate((emb_A1, emb_D1), axis=0)

    # batch_label_A1 = np.full(emb_A1.shape[0], 0)
    # batch_label_D1 = np.full(emb_D1.shape[0], 1)
    # batch_label = np.concatenate((batch_label_A1, batch_label_D1), axis=0)

    emb = np.load(
        '/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/yjxiao/COSMOS/results/HLN/Fusion/embedding_before_clustering.npy'
    )

    adata = sc.read_h5ad('/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/SpatialGlue_PRAGA/SpatialGlue/Results/HLN/SpatialGlue_HLN_BC.h5ad')

    batch_label = adata.obs['batch'].values
    # batch_label = batch_label[1:]

    print(emb.shape)

    print(batch_label.shape)
    #
    # assert 0

    adj_matrix = knn_adj_matrix(emb)

    eval_BC(emb, batch_label, adj_matrix, 'COSMOS', 'HLN_BC')

def HT():
    emb_0 = np.load(
        '/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/swruan/results/embedding/HT/Fusion/batch0_merged_embedding.npy')
    emb_1 = np.load(
        '/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/swruan/results/embedding/HT/Fusion/batch1_merged_embedding.npy')
    emb_2 = np.load(
        '/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/swruan/results/embedding/HT/Fusion/batch2_merged_embedding.npy')
    # emb = np.concatenate((emb_0, emb_1, emb_2), axis=0)

    emb = np.load(
        '/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/yjxiao/COSMOS/results/HT/Fusion/embedding_before_clustering.npy')

    print(emb.shape)

    batch_label_0 = np.full(emb_0.shape[0], 0)
    batch_label_1 = np.full(emb_1.shape[0], 1)
    batch_label_2 = np.full(emb_2.shape[0], 2)
    batch_label = np.concatenate((batch_label_0, batch_label_1, batch_label_2), axis=0)


    adj_matrix = knn_adj_matrix(emb)

    eval_BC(emb, batch_label, adj_matrix, 'COSMOS', 'HT_BC')

def MISAR_S1():
    emb_0 = np.load(
        '/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/swruan/results/embedding/MISAR/Fusion/batch0_merged_embedding.npy')
    emb_1 = np.load(
        '/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/swruan/results/embedding/MISAR/Fusion/batch1_merged_embedding.npy')
    emb_2 = np.load(
        '/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/swruan/results/embedding/MISAR/Fusion/batch2_merged_embedding.npy')
    emb_3 = np.load(
        '/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/swruan/results/embedding/MISAR/Fusion/batch3_merged_embedding.npy')
    # emb = np.concatenate((emb_0, emb_1, emb_2, emb_3), axis=0)
    emb = np.load('/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/yjxiao/COSMOS/results/MISAR/Fusion/embedding_before_clustering.npy')

    batch_label_0 = np.full(emb_0.shape[0], 0)
    batch_label_1 = np.full(emb_1.shape[0], 1)
    batch_label_2 = np.full(emb_2.shape[0], 2)
    batch_label_3 = np.full(emb_3.shape[0], 3)
    batch_label = np.concatenate((batch_label_0, batch_label_1, batch_label_2, batch_label_3), axis=0)

    adj_matrix = knn_adj_matrix(emb)

    eval_BC(emb, batch_label, adj_matrix, 'COSMOS', 'MISAR_S1_BC')

def MISAR_S2():
    emb_0 = np.load(
        '/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/swruan/results/embedding/MISARS2/Fusion/batch0_merged_embedding.npy')
    emb_1 = np.load(
        '/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/swruan/results/embedding/MISARS2/Fusion/batch1_merged_embedding.npy')
    emb_2 = np.load(
        '/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/swruan/results/embedding/MISARS2/Fusion/batch2_merged_embedding.npy')
    emb_3 = np.load(
        '/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/swruan/results/embedding/MISARS2/Fusion/batch3_merged_embedding.npy')
    # emb = np.concatenate((emb_0, emb_1, emb_2, emb_3), axis=0)
    emb = np.load('/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/yjxiao/COSMOS/results/MISAR_S2/Fusion/embedding_before_clustering.npy')

    batch_label_0 = np.full(emb_0.shape[0], 0)
    batch_label_1 = np.full(emb_1.shape[0], 1)
    batch_label_2 = np.full(emb_2.shape[0], 2)
    batch_label_3 = np.full(emb_3.shape[0], 3)
    batch_label = np.concatenate((batch_label_0, batch_label_1, batch_label_2, batch_label_3), axis=0)

    adj_matrix = knn_adj_matrix(emb)

    eval_BC(emb, batch_label, adj_matrix, 'COSMOS', 'MISAR_S2_BC')

def MB():

    emb = np.load('/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/yjxiao/COSMOS/results/Mouse_Brain/Fusion/embedding_before_clustering.npy')

    adata = sc.read_h5ad(
        '/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/SpatialGlue_PRAGA/SpatialGlue/Results/Mouse_Brain/SpatialGlue_MB_BC.h5ad')

    batch_label = adata.obs['batch'].values

    adj_matrix = knn_adj_matrix(emb)

    eval_BC(emb, batch_label, adj_matrix, 'COSMOS', 'MB_BC')

def MS():
    emb_A1 = np.load(
        '/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/swruan/results/embedding/Mouse_slpeen/Fusion/batch0_merged_embedding.npy')
    emb_D1 = np.load(
        '/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/swruan/results/embedding/Mouse_slpeen/Fusion/batch1_merged_embedding.npy')
    # emb = np.concatenate((emb_A1, emb_D1), axis=0)

    emb = np.load('/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/yjxiao/COSMOS/results/Mouse_slpeen/Fusion/embedding_before_clustering.npy')

    batch_label_A1 = np.full(emb_A1.shape[0], 0)
    batch_label_D1 = np.full(emb_D1.shape[0], 1)
    batch_label = np.concatenate((batch_label_A1, batch_label_D1), axis=0)

    adj_matrix = knn_adj_matrix(emb)

    eval_BC(emb, batch_label, adj_matrix, 'COSMOS', 'MS_BC')

def MT():
    # emb_0 = np.load(
    #     '/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/swruan/results/embedding/MISARS2/Fusion/batch0_merged_embedding.npy')
    # emb_1 = np.load(
    #     '/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/swruan/results/embedding/MISARS2/Fusion/batch1_merged_embedding.npy')
    # emb_2 = np.load(
    #     '/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/swruan/results/embedding/MISARS2/Fusion/batch2_merged_embedding.npy')
    # emb_3 = np.load(
    #     '/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/swruan/results/embedding/MISARS2/Fusion/batch3_merged_embedding.npy')
    # emb = np.concatenate((emb_0, emb_1, emb_2, emb_3), axis=0)

    # batch_label_0 = np.full(emb_0.shape[0], 0)
    # batch_label_1 = np.full(emb_1.shape[0], 1)
    # batch_label_2 = np.full(emb_2.shape[0], 2)
    # batch_label_3 = np.full(emb_3.shape[0], 3)
    # batch_label = np.concatenate((batch_label_0, batch_label_1, batch_label_2, batch_label_3), axis=0)

    emb = np.load('/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/yjxiao/COSMOS/results/Mouse_Thymus/Fusion/embedding_before_clustering.npy')

    adata = sc.read_h5ad(
        '/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/SpatialGlue_PRAGA/SpatialGlue/Results/Mouse_Thymus/SpatialGlue_MT_BC.h5ad')
    # sc.pp.filter_genes(adata, min_counts=1)
    # sc.pp.filter_cells(adata, min_counts=1)

    batch_label = adata.obs['batch'].values
    # batch_label = batch_label[1:]

    print(batch_label.shape)
    print(emb.shape)

    adj_matrix = knn_adj_matrix(emb)

    eval_BC(emb, batch_label, adj_matrix, 'COSMOS', 'MT_BC')


HLN()
HT()
MISAR_S1()
MISAR_S2()
MS()
MT()
MB()