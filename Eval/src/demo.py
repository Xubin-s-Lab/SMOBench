
import scib
import pandas as pd
from sklearn import metrics
import numpy as np
import umap


from src.clustering import *
from src.compute_metric import *

def eval(embeddings, adj_matrix, y_pred, y_GT, Method, dataset, coo):

    # # community detection
    # adj_matrix = knn_adj_matrix(embedding)
    # y_pred = RunLeiden(adj_matrix) ###y_pred  your predict label

    reducer = umap.UMAP()
    umap_coord = reducer.fit_transform(embeddings) ####umap coordinate need save it for visualization

    ###########calculate metric should use three times, depending on the dataset with ground truth or without ground truth
    y_test = y_GT ##### Ground Truth
    # batches = np.load('xx.npy') ###batch information

    # asw_celltype = silhouette_simple(embeddings, y_test)
    # graph_clisi = graph_clisi(adj_matrix, cell_labels)

    # ##with batch
    # asw_batch = silhouette_batch(embeddings, batches, y_test)
    # graph_ilisi = graph_ilisi(adj_matrix, batches)
    # iso_labels = get_isolated_labels(y_test, batches)
    # iso_f1 = isolated_labels_f1(y_test, y_pred, batches)
    # iso_asw = isolated_labels_asw(y_test, embeddings, batches)
    # kBET = kBET_from_knn_matrix(adj_matrix, batches, y_test)


    # Dataset with ground truth
    if y_test is not None:
        metrics_dict = {

            'asw_celltype':silhouette_simple(embeddings, y_test),
            'graph_clisi': graph_clisi(adj_matrix, y_test),
            'Graph Connectivity': graph_connectivity(embeddings, np.ravel(y_test)),
            'ARI': metrics.adjusted_rand_score(np.ravel(y_test), np.ravel(y_pred)),
            'NMI': metrics.normalized_mutual_info_score(np.ravel(y_test), np.ravel(y_pred)),
            'FMI': metrics.fowlkes_mallows_score(np.ravel(y_test), np.ravel(y_pred)),
            'Silhouette Coefficient': metrics.silhouette_score(embeddings, y_pred, metric='euclidean'),
            'Calinski-Harabaz Index': metrics.calinski_harabasz_score(embeddings, y_pred),
            'Davies-Bouldin Index': metrics.davies_bouldin_score(embeddings, y_pred),
            'Purity': purity(y_pred, y_test),
            'AMI': metrics.adjusted_mutual_info_score(y_test, y_pred),
            'Homogeneity': metrics.homogeneity_score(y_test, y_pred),
            'Completeness': metrics.completeness_score(y_test, y_pred),
            'V-measure': metrics.v_measure_score(y_test, y_pred),
            'F-measure': F_measure(y_pred, y_test),
            'Jaccard Index': jaccard(y_pred, y_test),
            'Dice Index': Dice(y_pred, y_test)
        }
        df = pd.DataFrame(list(metrics_dict.items()), columns=['Metric', f'{Method}_{dataset}'])
        df.to_csv(f'/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/eval/results/SpaMosaic/Mosaic_Integration/RNA/{Method}_{dataset}_cluster_metrics_with_GT.csv', index=False)



    # Dataset without ground truth
    Moran, Geary = Moran_Geary(coo, y_pred)
    metrics_dict = {
       # 'Graph Connectivity': graph_connectivity(embeddings, np.ravel(y_test)),
        'Silhouette Coefficient': metrics.silhouette_score(embeddings, y_pred, metric='euclidean'),
        'Calinski-Harabaz Index': metrics.calinski_harabasz_score(embeddings, y_pred),
        'Davies-Bouldin Index': metrics.davies_bouldin_score(embeddings, y_pred),
        'Moran Index': Moran.I,
        'Geary C': Geary.C,
    }
    df2 = pd.DataFrame(list(metrics_dict.items()), columns=['Metric', f'{Method}_{dataset}'])
    df2.to_csv(f'/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/eval/results/SpaMosaic/Mosaic_Integration/RNA/{Method}_{dataset}_cluster_metrics_wo_GT.csv', index=False)


def eval_BC(embeddings, batches, adj_matrix, Method, dataset):
    # Fused Dataset(batches) with ground truth
    metrics_dict = {
        'asw_batch':silhouette_no_group(embeddings, batches),
        'graph_ilisi':graph_ilisi(adj_matrix, batches),
        # 'iso_labels':get_isolated_labels(y_test, batches),
        # 'iso_f1': isolated_labels_f1(y_test, y_pred, batches),
        # 'iso_asw' : isolated_labels_asw(y_test, embeddings, batches),
        'kBET' : kBET_from_knn_matrix_no_label(adj_matrix, batches) ,
        # 'asw_celltype':silhouette_simple(embeddings, y_test),
        'graph_ilisi': graph_ilisi(adj_matrix, batches),

        'Graph Connectivity': graph_connectivity(embeddings, np.ravel(batches)),
        # 'ARI': metrics.adjusted_rand_score(np.ravel(y_test), np.ravel(y_pred)),
        # 'NMI': metrics.normalized_mutual_info_score(np.ravel(y_test), np.ravel(y_pred)),
        # 'FMI': metrics.fowlkes_mallows_score(np.ravel(y_test), np.ravel(y_pred)),
        # 'Silhouette Coefficient': metrics.silhouette_score(embeddings, y_pred, metric='euclidean'),
        # 'Calinski-Harabaz Index': metrics.calinski_harabasz_score(embeddings, y_pred),
        # 'Davies-Bouldin Index': metrics.davies_bouldin_score(embeddings, y_pred),
        # 'Purity': purity(y_pred, y_test),
        # 'AMI': metrics.adjusted_mutual_info_score(y_test, y_pred),
        # 'Homogeneity': metrics.homogeneity_score(y_test, y_pred),
        # 'Completeness': metrics.completeness_score(y_test, y_pred),
        # 'V-measure': metrics.v_measure_score(y_test, y_pred),
        # 'F-measure': F_measure(y_pred, y_test),
        # 'Jaccard Index': jaccard(y_pred, y_test),
        # 'Dice Index': Dice(y_pred, y_test)
    }
    df1 = pd.DataFrame(list(metrics_dict.items()), columns=['Metric', f'{Method}_{dataset}'])
    df1.to_csv(f'/home/hxl/Spa_Multi-omics/SpatialMultiomicsBenchmarking-main/eval/results/SpaMosaic/BC/{Method}_{dataset}_BC.csv', index=False)
