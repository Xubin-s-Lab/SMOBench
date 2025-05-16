import os
import pickle
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import scipy
# from .preprocess import pca
import matplotlib.pyplot as plt

#os.environ['R_HOME'] = '/scbio4/tools/R/R-4.0.3_openblas/R-4.0.3'

def split_adata_ob(ads, ad_ref, ob='obs', key='emb'):
    len_ads = [_.n_obs for _ in ads]
    if ob=='obsm':
        split_obsms = np.split(ad_ref.obsm[key], np.cumsum(len_ads[:-1]))
        for ad, v in zip(ads, split_obsms):
            ad.obsm[key] = v
    else:
        split_obs = np.split(ad_ref.obs[key].to_list(), np.cumsum(len_ads[:-1]))
        for ad, v in zip(ads, split_obs):
            ad.obs[key] = v

def find_peak_overlaps(query, key):
    q_seqname = np.array(query.get_seqnames())
    k_seqname = np.array(key.get_seqnames())
    q_start = np.array(query.get_start())
    k_start = np.array(key.get_start())
    q_width = np.array(query.get_width())
    k_width = np.array(key.get_width())
    q_end = q_start + q_width
    k_end = k_start + k_width

    q_index = 0
    k_index = 0
    overlap_index = [[] for i in range(len(query))]
    overlap_count = [0 for i in range(len(query))]

    while True:
        if q_index == len(query) or k_index == len(key):
            return overlap_index, overlap_count

        if q_seqname[q_index] == k_seqname[k_index]:
            if k_start[k_index] >= q_start[q_index] and k_end[k_index] <= q_end[q_index]:
                overlap_index[q_index].append(k_index)
                overlap_count[q_index] += 1
                k_index += 1
            elif k_start[k_index] < q_start[q_index]:
                k_index += 1
            else:
                q_index += 1
        elif q_seqname[q_index] < k_seqname[k_index]:
            q_index += 1
        else:
            k_index += 1


def peak_sets_alignment(adata_list, sep=(":", "-"), min_width=20, max_width=10000, min_gap_width=1,
                        peak_region= None):
    from genomicranges import GenomicRanges
    from iranges import IRanges
    from biocutils.combine import combine

    ## Peak merging
    gr_list = []
    for i in range(len(adata_list)):
        seq_names = []
        starts = []
        widths = []
        regions = adata_list[i].var_names if peak_region is None else adata_list[i].obs[peak_region]
        for region in regions:
            # print(region)
            # assert 0
            seq_names.append(region.split(sep[0])[0])
            if sep[0] == sep[1]:
                start, end = region.split(sep[0])[1:]
            else:
                start, end = region.split(sep[0])[1].split(sep[1])
            width = int(end) - int(start)
            starts.append(int(start))
            widths.append(width)
        gr = GenomicRanges(seqnames=seq_names, ranges=IRanges(starts, widths)).sort()
        peaks = [seqname + sep[0] + str(start) + sep[1] + str(end) for seqname, start, end in
                 zip(gr.get_seqnames(), gr.get_start(), gr.get_end())]
        adata_list[i] = adata_list[i][:, peaks]
        gr_list.append(gr)

    gr_combined = combine(*gr_list)
    gr_merged = gr_combined.reduce(min_gap_width=min_gap_width).sort()
    print("Peak merged")

    ## Peak filtering
    # filter by intesect
    overlap_index_list = []
    index = np.ones(len(gr_merged)).astype(bool)
    for gr in gr_list:
        overlap_index, overlap_count = find_peak_overlaps(gr_merged, gr)
        index = (np.array(overlap_count) > 0) * index
        overlap_index_list.append(overlap_index)
    # filter by width
    index = index * (gr_merged.get_width() > min_width) * (gr_merged.get_width() < max_width)
    gr_merged = gr_merged.get_subset(index)
    common_peak = [seqname + ":" + str(start) + "-" + str(end) for seqname, start, end in
                   zip(gr_merged.get_seqnames(), gr_merged.get_start(), gr_merged.get_end())]
    print("Peak filtered")

    ## Merge count matrix
    adata_merged_list = []
    for adata, overlap_index in zip(adata_list, overlap_index_list):
        overlap_index = [overlap_index[i] for i in range(len(index)) if index[i]]
        X = adata.X.tocsc()
        X_merged = scipy.sparse.hstack([scipy.sparse.csr_matrix(X[:, cur].sum(axis=1)) for cur in overlap_index])
        adata_merged_list.append(
            sc.AnnData(X_merged, obs=adata.obs, var=pd.DataFrame(index=common_peak), obsm=adata.obsm))
    print("Matrix merged")

    return adata_merged_list


def gene_sets_alignment(adata_list):
    # assert len(adata_list) > 1, "User needs to input more than two datasets"
    # for i in range(len(adata_list)):
    #     adata_list[i].var_names = [item.lower() for item in adata_list[i].var_names]
    #     adata_list[i].var_names_make_unique()
    # common_genes = np.intersect1d(adata_list[0].var_names, adata_list[1].var_names)
    # for i in range(2, len(adata_list)):
    #     common_genes = np.intersect1d(common_genes, adata_list[i].var_names)
    # for i in range(len(adata_list)):
    #     adata_list[i] = adata_list[i][:, common_genes]

    # 1. 检查输入有效性
    assert len(adata_list) > 1, "至少需要两个数据集"

    # 2. 统一基因名格式并去重
    for adata in adata_list:
        adata.var_names = [name.lower() for name in adata.var_names]
        adata.var_names_make_unique()
        if len(adata.var_names) != adata.shape[1]:
            raise ValueError("基因名处理导致维度不一致")

    # 3. 计算共有基因
    common_genes = set(adata_list[0].var_names)
    for adata in adata_list[1:]:
        common_genes.intersection_update(adata.var_names)
    common_genes = sorted(common_genes)

    if not common_genes:
        raise ValueError("无共有基因")

    # 4. 切片并修复维度
    for i in range(len(adata_list)):
        try:
            # 按名称切片
            adata_list[i] = adata_list[i][:, common_genes]
        except KeyError as e:
            # 缺失基因时按位置索引
            gene_indices = [adata_list[i].var_names.get_loc(gene) for gene in common_genes]
            adata_list[i] = adata_list[i][:, gene_indices]

        # 强制同步 var_names
        adata_list[i].var_names = common_genes

    return adata_list

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def clustering(adata, n_clusters=7, key='emb', add_key='SpatialGlue', method='mclust', start=0.1, end=3.0, increment=0.01, use_pca=False, n_comps=20):
    """\
    Spatial clustering based the latent representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    key : string, optional
        The key of the input representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'. 
    start : float
        The start value for searching. The default is 0.1. Only works if the clustering method is 'leiden' or 'louvain'.
    end : float 
        The end value for searching. The default is 3.0. Only works if the clustering method is 'leiden' or 'louvain'.
    increment : float
        The step size to increase. The default is 0.01. Only works if the clustering method is 'leiden' or 'louvain'.  
    use_pca : bool, optional
        Whether use pca for dimension reduction. The default is false.

    Returns
    -------
    None.

    """
    
    # if use_pca:
    #    adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)
    
    if method == 'mclust':
       if use_pca: 
          adata = mclust_R(adata, used_obsm=key + '_pca', num_cluster=n_clusters)
       else:
          adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters)
       adata.obs[add_key] = adata.obs['mclust']
    elif method == 'leiden':
       if use_pca: 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['leiden']
    elif method == 'louvain':
       if use_pca: 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['louvain']
       
def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number
    
    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float 
        The end value for searching.
    increment : float
        The step size to increase.
        
    Returns
    -------
    res : float
        Resolution.
        
    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res     

def plot_weight_value(alpha, label, modality1='mRNA', modality2='protein'):
  """\
  Plotting weight values
  
  """  
  import pandas as pd  
  
  df = pd.DataFrame(columns=[modality1, modality2, 'label'])  
  df[modality1], df[modality2] = alpha[:, 0], alpha[:, 1]
  df['label'] = label
  df = df.set_index('label').stack().reset_index()
  df.columns = ['label_SpatialGlue', 'Modality', 'Weight value']
  ax = sns.violinplot(data=df, x='label_SpatialGlue', y='Weight value', hue="Modality",
                split=True, inner="quart", linewidth=1, show=False)
  ax.set_title(modality1 + ' vs ' + modality2) 

  plt.tight_layout(w_pad=0.05)
  plt.show()     
