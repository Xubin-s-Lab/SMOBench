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

data_dir = '/home/users/nus/dmeng/scratch/spbench/Datasets/HT/'

output_dir = './results/HT/slice3'
os.makedirs(output_dir, exist_ok=True)


ad1_rna = sc.read_h5ad(join(data_dir, 'slice1/s1_adata_rna.h5ad'))
ad1_adt = sc.read_h5ad(join(data_dir, 'slice1/s1_adata_rna.h5ad'))

ad2_rna = sc.read_h5ad(join(data_dir, 'slice2/s2_adata_rna.h5ad'))
ad2_adt = sc.read_h5ad(join(data_dir, 'slice2/s2_adata_adt.h5ad'))

ad3_rna = sc.read_h5ad(join(data_dir, 'slice3/s3_adata_rna.h5ad'))
ad3_adt = sc.read_h5ad(join(data_dir, 'slice3/s3_adata_adt.h5ad'))



input_dict_1 = {
    'rna': [ad1_rna],
    'adt': [ad1_adt]
}

input_dict_2 = {
    'rna': [ad2_rna],
    'adt': [ad2_adt]
}

input_dict_3 = {
    'rna': [ad3_rna],
    'adt': [ad3_adt]
}

input_dict = input_dict_3

# ad1_rna = gene_sets_alignment(ad1_rna)

# input_dict = {
#     'rna': [ad1_rna, ad2_rna,ad3_rna],
#     'adt': [ad1_adt, ad2_adt,ad3_adt]
# }

# input_dict['rna'] = gene_sets_alignment(input_dict['rna'])
# input_dict['adt'] = gene_sets_alignment(input_dict['adt'])

input_key = 'dimred_bc'

RNA_preprocess(input_dict['rna'], batch_corr=True, favor='scanpy', n_hvg=5000, batch_key='src', key=input_key)
ADT_preprocess(input_dict['adt'], batch_corr=True, batch_key='src', key=input_key)

model = SpaMosaic(
    modBatch_dict=input_dict, input_key=input_key,
    batch_key='src', intra_knn=2, inter_knn=2, w_g=0.8, 
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

ad_embs = model.infer_emb(input_dict, emb_key='emb', final_latent_key='merged_emb')
ad_mosaic = sc.concat(ad_embs)
ad_mosaic = utls.get_umap(ad_mosaic, use_reps=['merged_emb'])

# 获取训练后GPU内存使用情况
info = pynvml.nvmlDeviceGetMemoryInfo(handle)
gpu_memory_usage_after = info.used / (1024 * 1024)  # 转换为 MB

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