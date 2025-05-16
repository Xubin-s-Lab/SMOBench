import os
import math
from time import time
from pathlib import Path

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
import scanpy as sc

from spaMultiVAE import SPAMULTIVAE


class Args(object):
    def __init__(self):
        # === 基础路径设置 ===
        self.base_data_dir = Path("/home/users/nus/dmeng/scratch/spbench/Datasets/Mouse_Thymus/Dataset3_Mouse_Thymus1")
        self.base_result_dir = Path("/home/users/nus/dmeng/scratch/spbench/yjxiao/spaVAE/src/spaMultiVAE/results/Mouse_Thymus/Dataset3_Mouse_Thymus1")

        # === 数据与输出文件路径 ===
        self.rna_file = self.base_data_dir / "adata_RNA.h5ad"
        self.adt_file = self.base_data_dir / "adata_ADT.h5ad"
        self.model_file = self.base_result_dir / "model.pt"
        self.final_latent_file = self.base_result_dir / "final_latent.npy"
        self.nonzero_mask_file = self.base_result_dir / "nonzero_mask.txt"

        # === 模型参数 ===
        self.select_genes = 0
        self.select_proteins = 0
        self.batch_size = "auto"
        self.maxiter = 5000
        self.train_size = 0.95
        self.patience = 200
        self.lr = 5e-3
        self.weight_decay = 1e-6
        self.gene_noise = 0
        self.protein_noise = 0
        self.dropoutE = 0
        self.dropoutD = 0
        self.encoder_layers = [128, 64]
        self.GP_dim = 2
        self.Normal_dim = 18
        self.gene_decoder_layers = [128]
        self.protein_decoder_layers = [128]
        self.init_beta = 10
        self.min_beta = 4
        self.max_beta = 25
        self.KL_loss = 0.025
        self.num_samples = 1
        self.fix_inducing_points = True
        self.inducing_point_steps = 19
        self.fixed_gp_params = False
        self.loc_range = 20.
        self.kernel_scale = 20.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


def check_nan_inf(data, name="data"):
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        print(f"{name} contains NaN or Inf values!")
        return True
    else:
        print(f"{name} is clean.")
    return False


def check_nan_in_tensor(tensor, name="tensor"):
    if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
        print(f"{name} contains NaN or Inf values!")
        return True
    return False


def main():
    args = Args()
    args.base_result_dir.mkdir(parents=True, exist_ok=True)

    # === 数据读取 ===
    adata11 = sc.read(args.rna_file)
    adata22 = sc.read(args.adt_file)

    x1 = adata11.X.toarray().astype('float64') if not isinstance(adata11.X, np.ndarray) else adata11.X.astype('float64')
    x2 = adata22.X.toarray().astype('float64') if not isinstance(adata22.X, np.ndarray) else adata22.X.astype('float64')
    loc = adata22.obsm['spatial'].astype('float64')

    # === 检查 NaN 和 Inf ===
    if check_nan_inf(x1, "x1") or check_nan_inf(x2, "x2"):
        raise ValueError("Input data contains NaN or Inf values.")

    # === 非零掩码过滤 ===
    nonzero_mask = (x2.sum(axis=1) != 0)
    x1 = x1[nonzero_mask]
    x2 = x2[nonzero_mask]
    loc = loc[nonzero_mask]
    np.savetxt(args.nonzero_mask_file, nonzero_mask.astype(int), fmt="%d")

    # === 自动设置 batch size ===
    if args.batch_size == "auto":
        if x1.shape[0] <= 1024:
            args.batch_size = 128
        elif x1.shape[0] <= 2048:
            args.batch_size = 256
        else:
            args.batch_size = 512
    else:
        args.batch_size = int(args.batch_size)

    # === 创建 AnnData 对象并归一化 ===
    adata1 = sc.AnnData(x1, dtype="float64")
    adata2 = sc.AnnData(x2, dtype="float64")

    # 对 RNA 和 ADT 数据进行 normalize_total 和 log1p 归一化
    sc.pp.normalize_total(adata1, target_sum=1e4)
    sc.pp.log1p(adata1)

    sc.pp.normalize_total(adata2, target_sum=1e4)
    sc.pp.log1p(adata2)

    # 保留归一化前的原始数据
    adata1.raw = adata1.copy()
    adata2.raw = adata2.copy()

    # 设置 size_factors = 1（跳过 VAE 内部归一化）
    adata1.obs["size_factors"] = 1.0


    # === 空间坐标归一化 ===
    scaler = MinMaxScaler()
    loc = scaler.fit_transform(loc) * args.loc_range

    eps = 1e-5
    initial_inducing_points = np.mgrid[0:(1+eps):(1./args.inducing_point_steps), 0:(1+eps):(1./args.inducing_point_steps)].reshape(2, -1).T * args.loc_range

    # === 背景建模（使用 log1p 输入数据） ===
    gm = GaussianMixture(n_components=2, covariance_type="diag", n_init=20).fit(x2)
    back_idx = np.argmin(gm.means_, axis=0)

    # 确保没有 NaN 或无穷值
    gm_means = np.expm1(gm.means_)
    gm_means = np.nan_to_num(gm_means, nan=1e-6)  # 防止后续计算出 NaN
    protein_log_back_mean = np.log(gm_means[back_idx, np.arange(adata2.n_vars)])

    gm_covariances = np.sqrt(gm.covariances_)
    gm_covariances = np.nan_to_num(gm_covariances, nan=1e-6)  # 防止无穷大
    protein_log_back_scale = gm_covariances[back_idx, np.arange(adata2.n_vars)]

    # === 构建模型 ===
    model = SPAMULTIVAE(
        gene_dim=adata1.n_vars, protein_dim=adata2.n_vars,
        GP_dim=args.GP_dim, Normal_dim=args.Normal_dim, dynamicVAE=False,
        encoder_layers=args.encoder_layers, gene_decoder_layers=args.gene_decoder_layers, protein_decoder_layers=args.protein_decoder_layers,
        gene_noise=args.gene_noise, protein_noise=args.protein_noise, encoder_dropout=args.dropoutE, decoder_dropout=args.dropoutD,
        fixed_inducing_points=args.fix_inducing_points, initial_inducing_points=initial_inducing_points,
        fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, N_train=adata1.n_obs,
        KL_loss=args.KL_loss, init_beta=args.init_beta, min_beta=args.min_beta, max_beta=args.max_beta,
        protein_back_mean=protein_log_back_mean, protein_back_scale=protein_log_back_scale,
        dtype=torch.float64, device=args.device
    )

    # === 模型训练或加载 ===
    if not args.model_file.exists():
        print("Training model...")
        t0 = time()
        model.train_model(
            pos=loc,
            gene_ncounts=adata1.X, gene_raw_counts=adata1.raw.X, gene_size_factors=adata1.obs["size_factors"].values,
            protein_ncounts=adata2.X, protein_raw_counts=adata2.raw.X,
            lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
            num_samples=args.num_samples, train_size=args.train_size,
            maxiter=args.maxiter, patience=args.patience,
            save_model=True, model_weights=args.model_file
        )
        print('Training time: %d seconds.' % int(time() - t0))
    else:
        print("Loading existing model...")
        model.load_model(args.model_file)

    # === 提取潜在表达 ===
    print("Extracting final latent representation...")
    final_latent = model.batching_latent_samples(
        X=loc, gene_Y=adata1.X, protein_Y=adata2.X, batch_size=args.batch_size
    )

    # 检查潜在变量是否有 NaN
    if check_nan_in_tensor(final_latent, "final_latent"):
        raise ValueError("Latent variable contains NaN or Inf values.")

    np.save(args.final_latent_file, final_latent)


if __name__ == "__main__":
    main()
