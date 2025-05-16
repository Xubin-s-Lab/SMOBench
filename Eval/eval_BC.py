
import argparse
import scanpy as sc

from src.clustering import RunLeiden, RunLouvain, knn_adj_matrix
from src.demo import eval, eval_BC

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

    emb = adata.obsm[args.key]

    print('Running Leiden clustering with KNN...')
    adj_matrix = knn_adj_matrix(emb)

    batches = adata.obs[args.batch_key]

    eval_BC(emb, batches, adj_matrix, args.Method, args.Dataset)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to modify global variable')
    parser.add_argument('--adata_path', type=str, help='adata_path')
    parser.add_argument('--key', type=str, help='key')
    parser.add_argument('--batch_key', type=str, help='batch_key')
    parser.add_argument('--GT_Path', type=str, default='', help='GT_Path')
    parser.add_argument('--Method', type=str, default='Test', help='GT_Path')
    parser.add_argument('--Dataset', type=str, default='Test', help='GT_Path')
    args = parser.parse_args()
    main(args)