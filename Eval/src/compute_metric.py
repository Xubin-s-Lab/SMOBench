from sklearn.metrics import adjusted_rand_score 
from sklearn.metrics import normalized_mutual_info_score,fowlkes_mallows_score
from sklearn import metrics
import collections
import numbers
from typing import Any, Mapping, Optional, TypeVar, Union
import anndata as ad
import h5py
import numpy as np
import scipy.sparse
from typing import Tuple
import pandas as pd
import scanpy as sc
import scipy.spatial
import sklearn.metrics
import sklearn.neighbors
import pysal.lib as ps
from sklearn.neighbors import NearestNeighbors
from anndata import AnnData
from scipy.sparse.csgraph import connected_components
from esda.moran import Moran
from esda.geary import Geary



Array = Union[np.ndarray, scipy.sparse.spmatrix]
BackedArray = Union[h5py.Dataset, ad._core.sparse_dataset.SparseDataset]
AnyArray = Union[Array, BackedArray]
ArrayOrScalar = Union[np.ndarray, numbers.Number]
Kws = Optional[Mapping[str, Any]]
RandomState = Optional[Union[np.random.RandomState, int]]

T = TypeVar("T")  # Generic type var

def Moran_Geary(coordinates, labels):
    n_neighbors = 3
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coordinates)  # +1 包含自身
    _, indices = nbrs.kneighbors(coordinates)

    # print(indices)
    # assert 0

    # 构建邻居字典
    neighbors_dict = {}
    for i in range(indices.shape[0]):
        neighbors_dict[i] = list(indices[i, 1:])  # 跳过自身，所以从 1 开始

    # 使用邻居字典构建空间权重矩阵
    w = ps.weights.W(neighbors_dict)

    w.transform = 'r'

    # 3. 计算 Moran's I score
    # 这里使用聚类标签作为输入变量
    moran_I = Moran(labels, w)

    # 计算 Geary's C
    geary_C = Geary(labels, w)

    return moran_I, geary_C

def get_rs(x: RandomState = None) -> np.random.RandomState:
    r"""
    Get random state object

    Parameters
    ----------
    x
        Object that can be converted to a random state object

    Returns
    -------
    rs
        Random state object
    """
    if isinstance(x, int):
        return np.random.RandomState(x)
    if isinstance(x, np.random.RandomState):
        return x
    return np.random
r"""
Performance evaluation metrics
"""


'''
---------------------
part GLUE fucntions
author:https://github.com/gao-lab/GLUE
MIT LICENSE
---------------------
'''


def mean_average_precision(
        x: np.ndarray, y: np.ndarray, neighbor_frac: float = 0.01, **kwargs
) -> float:
    r"""
    Mean average precision

    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    neighbor_frac
        Nearest neighbor fraction
    **kwargs
        Additional keyword arguments are passed to
        :class:`sklearn.neighbors.NearestNeighbors`

    Returns
    -------
    map
        Mean average precision
    """
    k = max(round(y.shape[0] * neighbor_frac), 1)
    nn = sklearn.neighbors.NearestNeighbors(
        n_neighbors=min(y.shape[0], k + 1), **kwargs
    ).fit(x)
    nni = nn.kneighbors(x, return_distance=False)
    match = np.equal(y[nni[:, 1:]], np.expand_dims(y, 1))
    return np.apply_along_axis(_average_precision, 1, match).mean().item()


def _average_precision(match: np.ndarray) -> float:
    if np.any(match):
        cummean = np.cumsum(match) / (np.arange(match.size) + 1)
        return cummean[match].mean().item()
    return 0.0


def normalized_mutual_info(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
    r"""
    Normalized mutual information with true clustering

    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.metrics.normalized_mutual_info_score`

    Returns
    -------
    nmi
        Normalized mutual information

    Note
    ----
    Follows the definition in `OpenProblems NeurIPS 2021 competition
    <https://openproblems.bio/neurips_docs/about_tasks/task3_joint_embedding/>`__
    """
    x = AnnData(X=x, dtype=x.dtype)
    sc.pp.neighbors(x, n_pcs=0, use_rep="X")
    nmi_list = []
    for res in (np.arange(20) + 1) / 10:
        sc.tl.leiden(x, resolution=res)
        leiden = x.obs["leiden"]
        nmi_list.append(sklearn.metrics.normalized_mutual_info_score(
            y, leiden, **kwargs
        ).item())
    return max(nmi_list)


def avg_silhouette_width(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
    r"""
    Cell type average silhouette width

    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.metrics.silhouette_score`

    Returns
    -------
    asw
        Cell type average silhouette width

    Note
    ----
    Follows the definition in `OpenProblems NeurIPS 2021 competition
    <https://openproblems.bio/neurips_docs/about_tasks/task3_joint_embedding/>`__
    """
    return (sklearn.metrics.silhouette_score(x, y, **kwargs).item() + 1) / 2


def graph_connectivity(
        x: np.ndarray, y: np.ndarray, **kwargs
) -> float:
    r"""
    Graph connectivity

    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`scanpy.pp.neighbors`

    Returns
    -------
    conn
        Graph connectivity
    """
    x = AnnData(X=x, dtype=x.dtype)
    sc.pp.neighbors(x, n_pcs=0, use_rep="X", **kwargs)
    conns = []
    for y_ in np.unique(y):
        x_ = x[y == y_]
        _, c = connected_components(
            x_.obsp['connectivities'],
            connection='strong'
        )
        counts = pd.value_counts(c)
        conns.append(counts.max() / counts.sum())
    return np.mean(conns).item()


def seurat_alignment_score(
        x: np.ndarray, y: np.ndarray, neighbor_frac: float = 0.01,
        n_repeats: int = 4, random_state: RandomState = None, **kwargs
) -> float:
    r"""
    Seurat alignment score

    Parameters
    ----------
    x
        Coordinates
    y
        Batch labels
    neighbor_frac
        Nearest neighbor fraction
    n_repeats
        Number of subsampling repeats
    random_state
        Random state
    **kwargs
        Additional keyword arguments are passed to
        :class:`sklearn.neighbors.NearestNeighbors`

    Returns
    -------
    sas
        Seurat alignment score
    """
    rs = get_rs(random_state)
    idx_list = [np.where(y == u)[0] for u in np.unique(y)]
    min_size = min(idx.size for idx in idx_list)
    repeat_scores = []
    for _ in range(n_repeats):
        subsample_idx = np.concatenate([
            rs.choice(idx, min_size, replace=False)
            for idx in idx_list
        ])
        subsample_x = x[subsample_idx]
        subsample_y = y[subsample_idx]
        k = max(round(subsample_idx.size * neighbor_frac), 1)
        nn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=k + 1, **kwargs
        ).fit(subsample_x)
        nni = nn.kneighbors(subsample_x, return_distance=False)
        same_y_hits = (
            subsample_y[nni[:, 1:]] == np.expand_dims(subsample_y, axis=1)
        ).sum(axis=1).mean()
        repeat_score = (k - same_y_hits) * len(idx_list) / (k * (len(idx_list) - 1))
        repeat_scores.append(min(repeat_score, 1))  # score may exceed 1, if same_y_hits is lower than expected by chance
    return np.mean(repeat_scores).item()


def avg_silhouette_width_batch(
        x: np.ndarray, y: np.ndarray, ct: np.ndarray, **kwargs
) -> float:
    r"""
    Batch average silhouette width

    Parameters
    ----------
    x
        Coordinates
    y
        Batch labels
    ct
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.metrics.silhouette_samples`

    Returns
    -------
    asw_batch
        Batch average silhouette width

    Note
    ----
    Follows the definition in `OpenProblems NeurIPS 2021 competition
    <https://openproblems.bio/neurips_docs/about_tasks/task3_joint_embedding/>`__
    """
    s_per_ct = []
    for t in np.unique(ct):
        mask = ct == t
        try:
            s = sklearn.metrics.silhouette_samples(x[mask], y[mask], **kwargs)
        except ValueError:  # Too few samples
            s = 0
        s = (1 - np.fabs(s)).mean()
        s_per_ct.append(s)
    return np.mean(s_per_ct).item()


def neighbor_conservation(
        x: np.ndarray, y: np.ndarray, batch: np.ndarray,
        neighbor_frac: float = 0.01, **kwargs
) -> float:
    r"""
    Neighbor conservation score

    Parameters
    ----------
    x
        Cooordinates after integration
    y
        Coordinates before integration
    b
        Batch
    **kwargs
        Additional keyword arguments are passed to
        :class:`sklearn.neighbors.NearestNeighbors`

    Returns
    -------
    nn_cons
        Neighbor conservation score
    """
    nn_cons_per_batch = []
    for b in np.unique(batch):
        mask = batch == b
        x_, y_ = x[mask], y[mask]
        k = max(round(x.shape[0] * neighbor_frac), 1)
        nnx = sklearn.neighbors.NearestNeighbors(
            n_neighbors=min(x_.shape[0], k + 1), **kwargs
        ).fit(x_).kneighbors_graph(x_)
        nny = sklearn.neighbors.NearestNeighbors(
            n_neighbors=min(y_.shape[0], k + 1), **kwargs
        ).fit(y_).kneighbors_graph(y_)
        nnx.setdiag(0)  # Remove self
        nny.setdiag(0)  # Remove self
        n_intersection = nnx.multiply(nny).sum(axis=1).A1
        n_union = (nnx + nny).astype(bool).sum(axis=1).A1
        nn_cons_per_batch.append((n_intersection / n_union).mean())
    return np.mean(nn_cons_per_batch).item()


def foscttm(
        x: np.ndarray, y: np.ndarray, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Fraction of samples closer than true match (smaller is better)

    Parameters
    ----------
    x
        Coordinates for samples in modality X
    y
        Coordinates for samples in modality y
    **kwargs
        Additional keyword arguments are passed to
        :func:`scipy.spatial.distance_matrix`

    Returns
    -------
    foscttm_x, foscttm_y
        FOSCTTM for samples in modality X and Y, respectively

    Note
    ----
    Samples in modality X and Y should be paired and given in the same order
    """
    if x.shape != y.shape:
        raise ValueError("Shapes do not match!")
    d = scipy.spatial.distance_matrix(x, y, **kwargs)
    foscttm_x = (d < np.expand_dims(np.diag(d), axis=1)).mean(axis=1)
    foscttm_y = (d < np.expand_dims(np.diag(d), axis=0)).mean(axis=0)
    return foscttm_x, foscttm_y


def purity(result, label):
    total_num = len(label)
    cluster_counter = collections.Counter(result)
    original_counter = collections.Counter(label)

    t = []
    for k in cluster_counter:
        p_k = []
        for j in original_counter:
            count = 0
            for i in range(len(result)):
                if result[i] == k and label[i] == j: # 求交集
                    count += 1
            p_k.append(count)
        temp_t = max(p_k)
        t.append(temp_t)
    
    return sum(t)/total_num
##F_value
def contingency_table(result, label):
    
    total_num = len(label)
    
    TP = TN = FP = FN = 0
    for i in range(total_num):
        for j in range(i + 1, total_num):
            if label[i] == label[j] and result[i] == result[j]:
                TP += 1
            elif label[i] != label[j] and result[i] != result[j]:
                TN += 1
            elif label[i] != label[j] and result[i] == result[j]:
                FP += 1
            elif label[i] == label[j] and result[i] != result[j]:
                FN += 1
    return (TP, TN, FP, FN)
def precision(result, label):
    TP, TN, FP, FN = contingency_table(result, label)
    return 1.0*TP/(TP + FP)

def recall(result, label):
    TP, TN, FP, FN = contingency_table(result, label)
    return 1.0*TP/(TP + FN)

def F_measure(result, label, beta=1):
    prec = precision(result, label)
    r = recall(result, label)
    return (beta*beta + 1) * prec * r/(beta*beta * prec + r)
###jaccard index
def jaccard(result,label):
    TP, TN, FP, FN = contingency_table(result, label)
    return TP/(TP + FP +FN)
###Dice index
def Dice(result,label):
    TP, TN, FP, FN = contingency_table(result, label)
    return 2*TP/(2*TP + FP +FN)



###isolated


import numpy as np
import pandas as pd
from sklearn.metrics.cluster import silhouette_samples, silhouette_score
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix
from numba import njit

# ========================
# Silhouette评分
# ========================
def silhouette_simple(X, labels, scale=True):
    """计算平均轮廓宽度"""
    asw = silhouette_score(X=X, labels=labels)
    if scale:
        asw = (asw + 1) / 2
    return asw


def silhouette_batch(X, batch_labels, group_labels, scale=True):
    """计算批次轮廓分数"""
    sil_per_label = []
    
    for group in np.unique(group_labels):
        group_idx = np.where(group_labels == group)[0]
        n_batches = len(np.unique(batch_labels[group_idx]))
        
        if n_batches <= 1 or n_batches == len(group_idx):
            continue
        
        sil = silhouette_samples(X[group_idx], batch_labels[group_idx])
        sil = np.abs(sil)
        
        if scale:
            sil = 1 - sil
            
        sil_per_label.extend([(group, s) for s in sil])
    
    if not sil_per_label:
        return np.nan
        
    sil_df = pd.DataFrame.from_records(sil_per_label, columns=["group", "score"])
    return sil_df.groupby("group").mean()["score"].mean()

def silhouette_no_group(X, batch_labels, scale=True):
    """计算不考虑分组的轮廓分数"""
    sil = silhouette_samples(X, batch_labels)
    sil = np.abs(sil)

    if scale:
        sil = 1 - sil

    return np.mean(sil) if sil.size > 0 else np.nan


# ========================
# LISI评分
# ========================
def graph_ilisi(knn_graph, batch_labels, scale=True):
    """计算图形整合LISI (iLISI)"""
    return _compute_lisi(knn_graph, batch_labels, scale, is_ilisi=True)


def graph_clisi(knn_graph, cell_labels, scale=True):
    """计算图形细胞类型LISI (cLISI)"""
    return _compute_lisi(knn_graph, cell_labels, scale, is_ilisi=False)


def _compute_lisi(knn_graph, labels, scale=True, is_ilisi=True):
    """LISI计算核心函数"""
    # 确保为csr矩阵
    if not isinstance(knn_graph, csr_matrix):
        knn_graph = csr_matrix(knn_graph)
    
    # 提取KNN信息
    n_cells = knn_graph.shape[0]
    n_neighbors = min(90, max(15, knn_graph.getnnz(axis=1).max()))
    perplexity = n_neighbors / 3
    
    # 类别编码
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    label_idx = np.array([label_to_idx[label] for label in labels])
    
    # 计算LISI
    lisi_values = []
    
    for i in range(n_cells):
        # 获取邻居和连接值
        idx_start, idx_end = knn_graph.indptr[i], knn_graph.indptr[i+1]
        if idx_start == idx_end:  # 没有邻居
            continue
            
        nn_indices = knn_graph.indices[idx_start:idx_end]
        nn_weights = knn_graph.data[idx_start:idx_end]
        
        if len(nn_indices) < 5:  # 邻居太少
            continue
            
        # 计算距离（从连接值）
        distances = -np.log(nn_weights)
        
        # 计算概率分布P
        beta = 1.0
        H, P = Hbeta(distances, beta)
        
        # 优化beta以匹配perplexity
        logU = np.log(perplexity)
        tries = 0
        
        while np.abs(H - logU) > 0.01 and tries < 20:
            if H > logU:
                beta *= 2
            else:
                beta /= 2
            H, P = Hbeta(distances, beta)
            tries += 1
        
        # 计算Simpson指数
        nn_labels = label_idx[nn_indices]
        one_hot = np.eye(n_labels)[nn_labels]
        sumP = np.matmul(P, one_hot)
        simpson = np.dot(sumP, sumP)
        
        # LISI = 1/Simpson
        lisi = 1.0 / simpson if simpson > 0 else np.nan
        lisi_values.append(lisi)
    
    # 计算中位数
    lisi_median = np.nanmedian(lisi_values)
    
    # 缩放
    if scale:
        if is_ilisi:  # iLISI: 值越大越好
            return (lisi_median - 1) / (n_labels - 1) if n_labels > 1 else 0
        else:  # cLISI: 值越小越好
            return (n_labels - lisi_median) / (n_labels - 1) if n_labels > 1 else 0
    
    return lisi_median


@njit
def Hbeta(distances, beta):
    """计算H和P"""
    P = np.exp(-distances * beta)
    sumP = np.sum(P)
    
    if sumP == 0:
        return 0, np.zeros_like(distances)
        
    H = np.log(sumP) + beta * np.sum(distances * P) / sumP
    P = P / sumP
    
    return H, P


# ========================
# 孤立标签评分
# ========================
def isolated_labels_f1(y_true, y_pred, batches, iso_threshold=None):
    """计算孤立标签的F1分数"""
    # 获取孤立标签
    isolated_labels = get_isolated_labels(y_true, batches, iso_threshold)
    
    if not isolated_labels:
        return None
    
    # 计算F1分数
    scores = []
    for label in isolated_labels:
        max_f1 = 0
        y_true_binary = (y_true == label)
        
        for cluster in np.unique(y_pred):
            y_pred_binary = (y_pred == cluster)
            f1 = f1_score(y_true_binary, y_pred_binary)
            max_f1 = max(max_f1, f1)
            
        scores.append(max_f1)
    
    return np.mean(scores)


def isolated_labels_asw(y_true, embeddings, batches, iso_threshold=None, scale=True):
    """计算孤立标签的ASW分数"""
    # 获取孤立标签
    isolated_labels = get_isolated_labels(y_true, batches, iso_threshold)
    
    if not isolated_labels:
        return None
    
    # 计算轮廓分数
    silhouette_values = silhouette_samples(embeddings, y_true)
    
    scores = []
    for label in isolated_labels:
        mask = y_true == label
        if np.sum(mask) > 0:
            score = np.mean(silhouette_values[mask])
            if scale:
                score = (score + 1) / 2
            scores.append(score)
    
    return np.mean(scores) if scores else None


def get_isolated_labels(labels, batches, iso_threshold=None):
    """获取孤立标签"""
    # 创建DataFrame并去重
    df = pd.DataFrame({'label': labels, 'batch': batches})
    tmp = df.drop_duplicates()
    
    # 计算每个标签的批次数
    batch_per_lab = tmp.groupby('label').agg({'batch': 'nunique'})
    
    # 确定阈值
    if iso_threshold is None:
        iso_threshold = batch_per_lab['batch'].min()
    
    if iso_threshold == len(np.unique(batches)):
        return []
    
    return batch_per_lab[batch_per_lab['batch'] <= iso_threshold].index.tolist()




from scipy.stats import chi2, chisquare
from scipy.sparse import csr_matrix
def kBET_from_knn_matrix(knn_graph, batch_labels, cell_labels, k_neighbors=50, scaled=True, verbose=False):
    """从KNN图计算kBET分数"""
    # 确保为CSR格式
    if not isinstance(knn_graph, csr_matrix):
        knn_graph = csr_matrix(knn_graph)
    
    # 提取KNN索引
    n_cells = knn_graph.shape[0]
    knn_indices = np.zeros((n_cells, k_neighbors), dtype=int)
    
    for i in range(n_cells):
        start, end = knn_graph.indptr[i], knn_graph.indptr[i+1]
        if start == end:  # 没有邻居
            knn_indices[i] = -1
            continue
        
        neighbors = knn_graph.indices[start:end]
        weights = knn_graph.data[start:end]
        
        # 按权重排序选择前k个
        if len(neighbors) > k_neighbors:
            top_k_idx = np.argsort(-weights)[:k_neighbors]
            knn_indices[i] = neighbors[top_k_idx]
        else:
            knn_indices[i, :len(neighbors)] = neighbors
            knn_indices[i, len(neighbors):] = -1
    
    # 计算kBET分数
    return kBET_score(knn_indices, batch_labels, cell_labels, scaled, verbose=verbose)


def kBET_score(knn_indices, batch_labels, cell_labels, scaled=True, return_df=False, verbose=False):
    """计算kBET批次混合分数"""
    # 将批次标签转换为数值索引
    unique_batches = np.unique(batch_labels)
    batch_to_idx = {label: idx for idx, label in enumerate(unique_batches)}
    batch_numeric = np.array([batch_to_idx[label] for label in batch_labels])
    n_batches = len(unique_batches)
    
    # 计算全局批次分布
    global_batch_dist = np.zeros(n_batches)
    for i in range(n_batches):
        global_batch_dist[i] = np.sum(batch_numeric == i) / len(batch_numeric)
    
    # 找出符合条件的标签
    counts = pd.DataFrame({'label': cell_labels, 'batch': batch_labels})
    counts = counts.groupby('label').agg({
        'label': 'count',
        'batch': lambda x: len(np.unique(x))
    })
    
    valid_labels = counts[(counts['label'] >= 10) & (counts['batch'] > 1)].index.tolist()
    skipped_labels = list(set(np.unique(cell_labels)) - set(valid_labels))
    
    if verbose:
        print(f"{len(skipped_labels)} 个标签只有单个批次或太小，已跳过。")
    
    # 准备kBET分数
    kBET_scores = {"cluster": skipped_labels, "kBET": [np.nan] * len(skipped_labels)}
    
    # 对每个有效的细胞类型计算kBET
    for clus in valid_labels:
        clus_idx = np.where(cell_labels == clus)[0]
        clus_batches = batch_numeric[clus_idx]
        
        # 计算k0（邻居数量）
        batch_counts = np.array([np.sum(clus_batches == i) for i in range(n_batches)])
        quarter_mean = np.floor(np.mean(batch_counts[batch_counts > 0]) / 4).astype(int)
        k0 = np.min([70, np.max([10, quarter_mean])])
        
        # 计算拒绝率
        rejection_rate = calculate_rejection_rate(
            knn_indices[clus_idx],
            batch_numeric,
            global_batch_dist,
            k0,
            verbose
        )
        
        kBET_scores["cluster"].append(clus)
        kBET_scores["kBET"].append(rejection_rate)
    
    # 转换为DataFrame
    kBET_df = pd.DataFrame.from_dict(kBET_scores)
    
    if return_df:
        return kBET_df
    
    # 计算最终分数
    final_score = np.nanmean(kBET_df["kBET"])
    
    # 缩放分数
    return 1 - final_score if scaled else final_score


def kBET_from_knn_matrix_no_label(knn_graph, batch_labels, k_neighbors=50, scaled=True, verbose=False):
    """
    从KNN图计算kBET分数，不考虑细胞类型标签。

    参数:
    knn_graph (csr_matrix): KNN图，邻接矩阵，行为细胞，列为k个最近邻。
    batch_labels (array-like): 每个细胞的批次标签。
    k_neighbors (int, 可选): 考虑的最近邻数量。默认为50。
    scaled (bool, 可选): 是否将kBET分数缩放到[0, 1]范围内。默认为True。
    verbose (bool, 可选): 是否打印详细信息。默认为False。

    返回:
    float: kBET分数。
    """
    # 确保为CSR格式
    if not isinstance(knn_graph, csr_matrix):
        knn_graph = csr_matrix(knn_graph)

    # 提取KNN索引
    n_cells = knn_graph.shape[0]
    knn_indices = np.zeros((n_cells, k_neighbors), dtype=int)

    for i in range(n_cells):
        start, end = knn_graph.indptr[i], knn_graph.indptr[i+1]
        if start == end:  # 没有邻居
            knn_indices[i] = -1
            continue

        neighbors = knn_graph.indices[start:end]
        weights = knn_graph.data[start:end]

        # 按权重排序选择前k个
        if len(neighbors) > k_neighbors:
            top_k_idx = np.argsort(-weights)[:k_neighbors]
            knn_indices[i] = neighbors[top_k_idx]
        else:
            knn_indices[i, :len(neighbors)] = neighbors
            knn_indices[i, len(neighbors):] = -1

    # 计算kBET分数
    return kBET_score_no_label(knn_indices, batch_labels, scaled, verbose=verbose)


def kBET_score_no_label(knn_indices, batch_labels, scaled=True, verbose=False):
    """
    计算kBET批次混合分数，不考虑细胞类型标签。

    参数:
    knn_indices (array-like): KNN索引矩阵，形状为 (n_cells, k_neighbors)。
    batch_labels (array-like): 每个细胞的批次标签。
    scaled (bool, 可选): 是否将kBET分数缩放到[0, 1]范围内。默认为True。
    verbose (bool, 可选): 是否打印详细信息。默认为False。

    返回:
    float: kBET分数。
    """
    n_cells = knn_indices.shape[0]
    n_batches = len(np.unique(batch_labels))

    # 将批次标签转换为数值索引
    unique_batches = np.unique(batch_labels)
    batch_to_idx = {label: idx for idx, label in enumerate(unique_batches)}
    batch_numeric = np.array([batch_to_idx[label] for label in batch_labels])

    # 计算全局批次分布
    global_batch_dist = np.zeros(n_batches)
    for i in range(n_batches):
        global_batch_dist[i] = np.sum(batch_numeric == i) / n_cells

    observed_batch_counts = np.zeros(n_batches)
    expected_batch_counts = np.zeros(n_batches)
    p_values = np.zeros(n_cells)
    degrees_of_freedom = n_batches - 1

    for cell_idx in range(n_cells):
        neighbors = knn_indices[cell_idx]
        valid_neighbors = neighbors[neighbors != -1]  # 排除无效邻居

        if len(valid_neighbors) == 0:
            p_values[cell_idx] = np.nan
            continue

        neighbor_batches = batch_numeric[valid_neighbors]

        # 统计观察到的邻居批次分布
        observed_batch_counts = np.array([np.sum(neighbor_batches == i) for i in range(n_batches)])
        # 计算期望的邻居批次分布
        expected_batch_counts = global_batch_dist * len(valid_neighbors)

        # 执行卡方检验
        _, p_value = chisquare(observed_batch_counts, expected_batch_counts)
        p_values[cell_idx] = p_value

    # 计算拒绝率
    rejection_rate = np.mean(p_values < 0.05)
    final_score = rejection_rate

    # 缩放分数
    if scaled:
        final_score = 1 - final_score

    return final_score


def calculate_rejection_rate(neighbors_idx, batch_labels, global_dist, k0, verbose=False):
    """计算kBET拒绝率"""
    n_cells = neighbors_idx.shape[0]
    n_batches = len(global_dist)
    rejection_count = 0
    total_tests = 0
    
    # 对每个细胞进行检验
    for i in range(n_cells):
        # 限制为k0个邻居
        k_limit = min(k0, neighbors_idx.shape[1])
        nn_idx = neighbors_idx[i, :k_limit]
        
        # 过滤无效的邻居索引
        valid_idx = nn_idx[nn_idx >= 0]
        
        if len(valid_idx) < 5:  # 邻居太少
            continue
        
        # 计算局部批次分布
        local_counts = np.zeros(n_batches)
        for b in range(n_batches):
            local_counts[b] = np.sum(batch_labels[valid_idx] == b)
        
        # 计算期望计数
        expected_counts = global_dist * len(valid_idx)
        
        # 进行卡方检验
        chi_sq_stat = 0
        dof = 0
        
        for b in range(n_batches):
            if expected_counts[b] > 0:
                chi_sq_stat += ((local_counts[b] - expected_counts[b]) ** 2) / expected_counts[b]
                dof += 1
        
        if dof > 1:
            dof -= 1  # 自由度为有效批次数-1
            p_val = 1 - chi2.cdf(chi_sq_stat, dof)
            
            # 如果p值小于0.05，则拒绝原假设
            if p_val < 0.05:
                rejection_count += 1
            
            total_tests += 1
    
    # 计算拒绝率
    if total_tests > 0:
        return rejection_count / total_tests
    else:
        return np.nan



