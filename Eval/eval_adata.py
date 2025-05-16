
import argparse
import scanpy as sc

from src.clustering import RunLeiden, RunLouvain, knn_adj_matrix
from src.demo import eval

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

def main(args):
    adata = sc.read_h5ad(args.adata_path)
    print(adata)

    if args.npy_path != '':
        emb = np.load(args.npy_path)
    else:
        emb = adata.obsm[args.key]

    adj_matrix = knn_adj_matrix(emb)

    if args.cluster_Path!='':
        Leiden_labels_list = read_list_from_file(args.cluster_Path)
    else:
        print('Running Leiden clustering with KNN...')
        Leiden_labels_list = RunLeiden(adj_matrix)
        print('Clustering Done!')

    if args.GT_Path=='':
        GT_Label_list = None
        print('Warning: No GT is provided!')
    else:
        GT_Label_list = read_list_from_file(args.GT_Path)

    eval(emb, adj_matrix, Leiden_labels_list, GT_Label_list, Method=args.Method, dataset=args.Dataset, coo=adata.obsm['spatial'])




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to modify global variable')
    parser.add_argument('--adata_path', type=str, help='adata_path')
    parser.add_argument('--npy_path', type=str, default='', help='npy_path')
    parser.add_argument('--key', type=str, default='', help='key')
    parser.add_argument('--GT_Path', type=str, default='', help='GT_Path')
    parser.add_argument('--cluster_Path', type=str, default='', help='GT_Path')
    parser.add_argument('--Method', type=str, default='Test', help='GT_Path')
    parser.add_argument('--Dataset', type=str, default='Test', help='GT_Path')
    args = parser.parse_args()
    main(args)