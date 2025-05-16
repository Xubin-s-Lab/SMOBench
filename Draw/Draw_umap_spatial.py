
import argparse
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from src.clustering import RunLeiden, RunLouvain, knn_adj_matrix

def read_list_from_file(path):
    list = []
    # 打开文件进行读取，使用 'r' 模式
    with open(path, 'r') as f:
        # 遍历文件中的每一行，将其转换为整数并添加到列表中
        for line in f:
            # 去掉行末的换行符，然后将字符串转换为整数
            num = int(line.strip())
            list.append(num)

    return list

def draw_spatial(adata, s, color, Save_name, title):
    fig = sc.pl.embedding(adata, basis='spatial', color=color, title=title, s=s, show=False, return_fig=True)
    # 获取当前的轴对象
    ax = fig.axes[0]

    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.savefig(Save_name, dpi=300)
    print('Saved figure to {}'.format(Save_name))

def draw_umap(adata, key, color, title, s, Save_name):
    sc.pp.neighbors(adata, use_rep=key, n_neighbors=30)
    sc.tl.umap(adata)
    # 绘制UMAP并指定自定义颜色，返回fig和ax对象以便进一步调整
    fig = sc.pl.umap(adata, color=color, title=title, s=s,
                     show=False, return_fig=True)

    # 获取当前的轴对象
    ax = fig.axes[0]

    # 移除横纵坐标名称
    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.savefig(Save_name, dpi=300)

def main(args):
    ## if Adata
    # adata = sc.read_h5ad(args.adata_path)
    # ## if npy
    adata = sc.read_h5ad(args.adata_path)
    emb = np.load(args.npy_path)
    adata.obsm[args.key] = emb

    print(adata)

    emb = adata.obsm[args.key]

    print('Running Leiden clustering with KNN...')
    adj_matrix = knn_adj_matrix(emb)
    Leiden_cluster = RunLeiden(adj_matrix)
    print('Clustering Done!')
    adata.obs['Leiden_cluster'] = np.array(Leiden_cluster).astype(str)
    adata.obs['Leiden_cluster'] = adata.obs['Leiden_cluster'].astype('category')

    output_cluster_label = args.Method + '_' + args.Dataset + '_Leiden_cluster.txt'

    with open(output_cluster_label, 'w') as f:
        for num in Leiden_cluster:
            f.write(f"{num}\n")
    print('Saving clustering label Done!')

    if args.GT_Path=='':
        GT_Label_list = None
        print('Warning: No GT is provided!')
    else:
        GT_Label_list = read_list_from_file(args.GT_Path)
        adata.obs['GT_Label'] = np.array(GT_Label_list).astype(str)
        adata.obs['GT_Label'] = adata.obs['GT_Label'].astype('category')

        ## Draw Umap with cluter
        Save_name = args.Method + '_' + args.Dataset + '_GT_umap.png'
        draw_umap(adata, key=args.key, color='GT_Label', title=args.Method+'_GT', s=args.size, Save_name=Save_name)

    ## Draw Umap with cluter
    Save_name = args.Method + '_' + args.Dataset + '_Leiden_cluster_umap.png'
    draw_umap(adata, key=args.key, color='Leiden_cluster', title=args.Method+'_Leiden_cluster', s=args.size, Save_name=Save_name)

    Save_name = args.Method + '_' + args.Dataset + '_Leiden_cluster_spatial.png'

    draw_spatial(adata, s=args.size, color='Leiden_cluster', Save_name=Save_name, title=args.Method+'_Leiden_cluster')







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to modify global variable')
    parser.add_argument('--adata_path', type=str, help='adata_path')
    parser.add_argument('--npy_path', type=str, default='', help='adata_path')
    parser.add_argument('--key', type=str, help='key')
    parser.add_argument('--GT_Path', type=str, default='', help='GT_Path')
    parser.add_argument('--size', type=float, help='size')
    parser.add_argument('--Method', type=str, default='Test', help='Method')
    parser.add_argument('--Dataset', type=str, default='Test', help='Dataset')
    args = parser.parse_args()
    main(args)