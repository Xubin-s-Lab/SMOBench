import numpy as np
import scanpy as sc
import subprocess
import os
import time
from COSMOS import cosmos
from COSMOS.pyWNN import pyWNN


# 定义路径变量
result_path = '/home/users/nus/dmeng/scratch/spbench/yjxiao/COSMOS/results/HLN/Fusion'

# 确保路径存在
if not os.path.exists(result_path):
    os.makedirs(result_path)

# 获取 GPU 利用率
def get_gpu_utilization():
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    gpu_utilization = result.stdout.decode('utf-8').strip()
    return gpu_utilization

# 运行时的日志文件保存
def save_logs(adata1, adata2, random_seed):
    # 记录开始时间
    start_time = time.time()

    # 运行 COSMOS
    cosmos_comb = cosmos.Cosmos(adata1=adata1, adata2=adata2)
    cosmos_comb.preprocessing_data(n_neighbors=10)
    cosmos_comb.train(
        spatial_regularization_strength=0.05, 
        z_dim=50,
        lr=1e-3, 
        wnn_epoch=500, 
        total_epoch=1000, 
        max_patience_bef=10, 
        max_patience_aft=30, 
        min_stop=200, 
        random_seed=random_seed, 
        gpu=0, 
        regularization_acceleration=True, 
        edge_subset_sz=1000000
    )

    # 记录结束时间
    end_time = time.time()
    running_time = end_time - start_time
    np.save(os.path.join(result_path, 'running_time.npy'), running_time)
    print(f"Total running time: {running_time} seconds")

    # 保存 Embedding
    embedding_before_clustering = cosmos_comb.embedding
    np.save(os.path.join(result_path, 'embedding_before_clustering.npy'), embedding_before_clustering)

    # 保存 Reconstructed Matrix (如果存在)
    if hasattr(cosmos_comb, 'reconstructed_matrix'):
        reconstructed_matrix = cosmos_comb.reconstructed_matrix
        np.save(os.path.join(result_path, 'reconstructed_matrix.npy'), reconstructed_matrix)
    else:
        print("Reconstructed Matrix does not exist.")

# 数据读取
adata1 = sc.read('/home/users/nus/dmeng/scratch/spbench/yjxiao/COSMOS/results/HLN/Fusion/adata1.h5ad')
adata2 = sc.read('/home/users/nus/dmeng/scratch/spbench/yjxiao/COSMOS/results/HLN/Fusion/adata2.h5ad')

# 设置随机种子
random_seed = 20

# 调用保存日志函数
save_logs(adata1, adata2, random_seed)
