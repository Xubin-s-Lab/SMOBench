import os
import scanpy as sc
from os.path import join

import numpy as np
import sys
import pynvml 
import time


import sys
sys.path.insert(0, '/home/users/nus/dmeng/scratch/spbench/swruan/SpaMosaic-main')

from spamosaic.framework import SpaMosaic

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # for CuBLAS operation and you have CUDA >= 10.2

import spamosaic.utils as utls
from spamosaic.preprocessing2 import RNA_preprocess, ADT_preprocess, Epigenome_preprocess

from Human_Lymph_Node.utils import gene_sets_alignment,peak_sets_alignment


# 设置数据目录
data_dir = '/home/users/nus/dmeng/scratch/spbench/Datasets/Mouse_Brain'



ad1_rna = sc.read_h5ad(join(data_dir, 'Dataset7_Mouse_Brain_ATAC/adata_RNA.h5ad'))
ad1_epi = sc.read_h5ad(join(data_dir, 'Dataset7_Mouse_Brain_ATAC/adata_peaks_normalized.h5ad'))

ad2_rna = sc.read_h5ad(join(data_dir, 'Dataset8_Mouse_Brain_H3K4me3/adata_RNA.h5ad'))
ad2_epi = sc.read_h5ad(join(data_dir, 'Dataset8_Mouse_Brain_H3K4me3/adata_peaks_normalized.h5ad'))

ad3_rna = sc.read_h5ad(join(data_dir, 'Dataset9_Mouse_Brain_H3K27ac/adata_RNA.h5ad'))
ad3_epi = sc.read_h5ad(join(data_dir, 'Dataset9_Mouse_Brain_H3K27ac/adata_peaks_normalized.h5ad'))

ad4_rna = sc.read_h5ad(join(data_dir, 'Dataset10_Mouse_Brain_H3K27me3/adata_RNA.h5ad'))
ad4_epi = sc.read_h5ad(join(data_dir, 'Dataset10_Mouse_Brain_H3K27me3/adata_peaks_normalized.h5ad'))



input_dict_1 = {
    'rna':     [ad1_rna],
    'atac':    [ad1_epi]
}

input_dict_2 = {
    'rna':     [ad2_rna],
    'atac':    [ad2_epi]
}

input_dict_3 = {
    'rna':     [ad3_rna],
    'atac':    [ad3_epi]
}

input_dict_4 = {
    'rna':     [ad4_rna],
    'atac':    [ad4_epi]
}


input_dict = {
    'rna':     [ad1_rna, ad2_rna, ad3_rna,ad4_rna],
    'atac':    [ad1_epi, ad2_epi, ad3_epi,ad4_epi]
}

input_key = 'dimred_bc'


input_dict = input_dict_4
output_dir = './results/Mouse_Brain/Dataset10'
os.makedirs(output_dir, exist_ok=True)

# for key in input_dict.keys():
#     if isinstance(input_dict[key], list):  # 确保该键对应的值是列表
#         for i, ad in enumerate(input_dict[key]):
#             ad.obs_names = f"s{i}_" + ad.obs_names


# input_dict['rna'] = gene_sets_alignment(input_dict['rna'])
# input_dict['atac'] = peak_sets_alignment(input_dict['atac'])


RNA_preprocess(input_dict['rna'], batch_corr=True, n_hvg=5000, batch_key='src', key=input_key)
Epigenome_preprocess(input_dict['atac'], batch_corr=True, n_peak=50000, batch_key='src', key=input_key)



model = SpaMosaic(
    modBatch_dict=input_dict, input_key=input_key,
    batch_key='src', intra_knn=10, inter_knn=10, w_g=0.8, 
    seed=1234, 
    device='cuda:0'
)

# 初始化GPU内存监控
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
info = pynvml.nvmlDeviceGetMemoryInfo(handle)
gpu_memory_usage_before = info.used / (1024 * 1024)  # 转换为 MB

start_time = time.time()

model.train(net='wlgcn', lr=0.01, T=0.01, n_epochs=100)


# 获取训练后GPU内存使用情况
info = pynvml.nvmlDeviceGetMemoryInfo(handle)
gpu_memory_usage_after = info.used / (1024 * 1024)  # 转换为 MB


ad_embs = model.infer_emb(input_dict, emb_key='emb', final_latent_key='merged_emb')
ad_mosaic = sc.concat(ad_embs)
ad_mosaic = utls.get_umap(ad_mosaic, use_reps=['merged_emb'])


# 计算运行时间
running_time = time.time() - start_time

# 保存嵌入
for k in model.mod_list:
    for bi in range(model.n_batches):
        if input_dict[k][bi] is not None:
            emb = input_dict[k][bi].obsm['emb']
            np.save(os.path.join(output_dir, f'{k}_batch{bi}_embedding.npy'), emb)

# 保存合并嵌入
for bi, ad in enumerate(ad_embs):
    np.save(os.path.join(output_dir, f'batch{bi}_merged_embedding.npy'), ad.obsm['merged_emb'])

# 使用Leiden聚类
sc.tl.leiden(ad_mosaic, resolution=1.0, key_added='leiden')  # 使用Leiden聚类[^1^]
np.save(os.path.join(output_dir, 'leiden_y_pred_label.npy'), ad_mosaic.obs['leiden'].values)


# 保存运行时间和GPU内存使用情况
with open(os.path.join(output_dir, 'running_time_and_memory.txt'), 'w') as f:
    f.write(f"Running Time: {running_time:.2f} seconds\n")
    f.write(f"GPU Memory Usage Before: {gpu_memory_usage_before:.2f} MB\n")
    f.write(f"GPU Memory Usage After: {gpu_memory_usage_after:.2f} MB\n")

print("end")