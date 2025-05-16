from typing import Optional
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sklearn
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import sklearn.decomposition
import sklearn.feature_extraction.text
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.utils.extmath
from harmony import harmonize
import anndata as ad
import pandas as pd

from spamosaic.utils import split_adata_ob
from scipy.sparse import vstack

class tfidfTransformer:
    def __init__(self):
        self.idf = None
        self.fitted = False

    def fit(self, X):
        self.idf = X.shape[0] / (1e-8+X.sum(axis=0))
        self.fitted = True

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Transformer was not fitted on any data")
        if scipy.sparse.issparse(X):
            tf = X.multiply(1 / (1e-8+X.sum(axis=1)))
            return tf.multiply(self.idf)
        else:
            tf = X / (1e-8+X.sum(axis=1, keepdims=True))
            return tf * self.idf

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# optional, other reasonable preprocessing steps also ok
class lsiTransformer:
    def __init__(
        self, n_components: int = 20, drop_first=True, use_highly_variable=None, log=True, norm=True, z_score=True,
        tfidf=True, svd=True, use_counts=False, pcaAlgo='arpack'
    ):  

        self.drop_first = drop_first
        self.n_components = n_components + drop_first
        self.use_highly_variable = use_highly_variable

        self.log = log
        self.norm = norm
        self.z_score = z_score
        self.svd = svd
        self.tfidf = tfidf
        self.use_counts = use_counts

        self.tfidfTransformer = tfidfTransformer()
        self.normalizer = sklearn.preprocessing.Normalizer(norm="l1")
        self.pcaTransformer = sklearn.decomposition.TruncatedSVD(
            n_components=self.n_components, random_state=777, algorithm=pcaAlgo
        )
        self.fitted = None

    def fit(self, adata: anndata.AnnData):
        if self.use_highly_variable is None:
            self.use_highly_variable = "highly_variable" in adata.var
        adata_use = (
            adata[:, adata.var["highly_variable"]]
            if self.use_highly_variable
            else adata
        )
        if self.use_counts:
            X = adata_use.layers['counts']
        else:
            X = adata_use.X
        if self.tfidf:
            X = self.tfidfTransformer.fit_transform(X)
        if scipy.sparse.issparse(X):
            X = X.A.astype("float32")
        if self.norm:
            X = self.normalizer.fit_transform(X)
        if self.log:
            X = np.log1p(X * 1e4)    # L1-norm and target_sum=1e4 and log1p
        self.pcaTransformer.fit(X)
        self.fitted = True

    def transform(self, adata):
        if not self.fitted:
            raise RuntimeError("Transformer was not fitted on any data")
        adata_use = (
            adata[:, adata.var["highly_variable"]]
            if self.use_highly_variable
            else adata
        )
        if self.use_counts:
            X_pp = adata_use.layers['counts']
        else:
            X_pp = adata_use.X
        if self.tfidf:
            X_pp = self.tfidfTransformer.transform(X_pp)
        if scipy.sparse.issparse(X_pp):
            X_pp = X_pp.A.astype("float32")
        if self.norm:
            X_pp = self.normalizer.transform(X_pp)
        if self.log:
            X_pp = np.log1p(X_pp * 1e4)
        if self.svd:
            X_pp = self.pcaTransformer.transform(X_pp)
        if self.z_score:
            X_pp -= X_pp.mean(axis=1, keepdims=True)
            X_pp /= (1e-8+X_pp.std(axis=1, ddof=1, keepdims=True))
        pp_df = pd.DataFrame(X_pp, index=adata_use.obs_names).iloc[
            :, int(self.drop_first) :
        ]
        return pp_df

    def fit_transform(self, adata):
        self.fit(adata)
        return self.transform(adata)
   
# CLR-normalization     
def clr_normalize(adata):
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    # sc.pp.pca(adata, n_comps=min(50, adata.n_vars-1))
    return adata

def harmony(latent, batch_labels, use_gpu=True):
    df_batches = pd.DataFrame(np.reshape(batch_labels, (-1, 1)), columns=['batch'])
    bc_latent = harmonize(
        latent, df_batches, batch_key="batch", use_gpu=use_gpu, verbose=True
    )
    return bc_latent

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

def combine_BC(adata_list):
    X_list = []
    spatial_list = []
    batch_list = []
    feat_list = []

    for i, adata in enumerate(adata_list):
        # print(adata.obsm['feat'].shape)
        # X_list.append(adata.obsm['feat'])
        print(adata.X.shape)
        X_list.append(adata.X)
        # X_list.append(adata.X.toarray())
        spatial_list.append(adata.obsm['spatial'])
        # feat_list.append(adata.obsm['feat'])
        batch_list.append(np.full(adata.n_obs, i))
    
    X_combined = vstack(X_list)
    # X_combined = np.concatenate(X_list, axis=0)

    spatial_combined = np.concatenate(spatial_list, axis=0)

    # feat_combined = np.concatenate(feat_list, axis=0)

    batch_combined = np.concatenate(batch_list, axis=0)

    adata_combine = ad.AnnData(X=X_combined)
    adata_combine.obsm['spatial'] = spatial_combined

    # 将 batch_combined 转为 Pandas Series，再转为 'category' 类型
    adata_combine.obs['src'] = batch_combined
    # 假设 batch_array 是 NumPy 数组
    batch_array = adata_combine.obs['src'].values  # 或从其他来源获取

    # 转换为分类类型
    adata_combine.obs['src'] = pd.Series(
        batch_array,
        dtype='category',
        index=adata_combine.obs.index  # 确保索引一致
    )

    # print(adata_combine.obs['src'])

    return adata_combine

def RNA_preprocess(rna_ads, batch_corr=False, favor='adapted', n_hvg=3000, lognorm=True, scale=False, batch_key='src', key='dimred_bc', return_hvf=False):
    measured_ads = [ad for ad in rna_ads if ad is not None]
    
    ad_concat = combine_BC(measured_ads)

    # 检查批次列是否存在空值
    # print(ad_concat.obs[batch_key].isnull().sum())
    #
    # assert 0

    if favor=='scanpy':
        if lognorm:
            sc.pp.normalize_total(ad_concat, target_sum=1e4)
            sc.pp.log1p(ad_concat)
        if n_hvg:
            # sc.pp.highly_variable_genes(ad_concat, n_top_genes=n_hvg)
            sc.pp.highly_variable_genes(ad_concat, n_top_genes=n_hvg, batch_key=batch_key)
            ad_concat = ad_concat[:, ad_concat.var.query('highly_variable').index.to_numpy()].copy()
        if scale: 
            sc.pp.scale(ad_concat)
        sc.pp.pca(ad_concat, n_comps=min(50, ad_concat.n_vars-1))
        tmp_key = 'X_pca'
    else:
        n_hvg = n_hvg if n_hvg else ad_concat.shape[1]
        sc.pp.highly_variable_genes(ad_concat, flavor='seurat_v3', n_top_genes=n_hvg, batch_key=batch_key)
        transformer = lsiTransformer(n_components=50, drop_first=False, log=True, norm=True, z_score=True, tfidf=False, svd=True, pcaAlgo='arpack')
        ad_concat.obsm['X_lsi'] = transformer.fit_transform(ad_concat[:, ad_concat.var.query('highly_variable').index.to_numpy()]).values
        tmp_key = 'X_lsi'
    
    if len(measured_ads) > 1 and batch_corr:
        ad_concat.obsm[key] = harmony(
            ad_concat.obsm[tmp_key], 
            ad_concat.obs[batch_key].to_list(), 
            use_gpu=True
        )
    else:
        ad_concat.obsm[key] = ad_concat.obsm[tmp_key]
    split_adata_ob([ad for ad in rna_ads if ad is not None], ad_concat, ob='obsm', key=key)

    if n_hvg and return_hvf:
        return ad_concat.var.query('highly_variable').index.to_numpy(), np.where(ad_concat.var['highly_variable'])[0]

def ADT_preprocess(adt_ads, batch_corr=False, favor='clr', lognorm=True, scale=False, batch_key='src', key='dimred_bc'):
    measured_ads = [ad for ad in adt_ads if ad is not None]
    # ad_concat = sc.concat(measured_ads)
    ad_concat = combine_BC(measured_ads)
    if favor=='clr':
        ad_concat = clr_normalize(ad_concat)
        # if scale: sc.pp.scale(ad_concat)
    else:
        if lognorm:
            sc.pp.normalize_total(ad_concat, target_sum=1e4)
            sc.pp.log1p(ad_concat)
        if scale: sc.pp.scale(ad_concat)
            
    sc.pp.pca(ad_concat, n_comps=min(50, ad_concat.n_vars-1))

    if len(measured_ads) > 1 and batch_corr:
        ad_concat.obsm[key] = harmony(ad_concat.obsm['X_pca'], ad_concat.obs[batch_key].to_list(), use_gpu=True)
    else:
        ad_concat.obsm[key] = ad_concat.obsm['X_pca']
    split_adata_ob([ad for ad in adt_ads if ad is not None], ad_concat, ob='obsm', key=key)

def Epigenome_preprocess(epi_ads, batch_corr=False, n_peak=100000, batch_key='src', key='dimred_bc', return_hvf=False):
    measured_ads = [ad for ad in epi_ads if ad is not None]
    # ad_concat = sc.concat(measured_ads)
    ad_concat = combine_BC(measured_ads)
    sc.pp.highly_variable_genes(ad_concat, flavor='seurat_v3', n_top_genes=n_peak, batch_key=batch_key)

    transformer = lsiTransformer(n_components=50, drop_first=True, log=True, norm=True, z_score=True, tfidf=True, svd=True, pcaAlgo='arpack')
    ad_concat.obsm['X_lsi'] = transformer.fit_transform(ad_concat[:, ad_concat.var.query('highly_variable').index.to_numpy()]).values

    if len(measured_ads) > 1 and batch_corr:
        ad_concat.obsm[key] = harmony(ad_concat.obsm['X_lsi'], ad_concat.obs[batch_key].to_list(), use_gpu=True)
    else:
        ad_concat.obsm[key] = ad_concat.obsm['X_lsi']
    
    split_adata_ob([ad for ad in epi_ads if ad is not None], ad_concat, ob='obsm', key=key)

    if return_hvf:
        return ad_concat.var.query('highly_variable').index.to_numpy(), np.where(ad_concat.var['highly_variable'])[0]






