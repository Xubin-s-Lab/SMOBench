
import scib
import pandas as pd
from sklearn import metrics
import numpy as np
import umap


from clustering import *
from compute_metric import *


embeddings = np.load('xx.npy')

# community detection
adj_matrix = knn_adj_matrix(embedding)
y_pred = RunLeiden(adj_matrix) ###y_pred  your predict label

reducer = umap.UMAP()
umap_coord = reducer.fit_transform(fin) ####umap coordinate need save it for visualization

###########calculate metric should use twice, depending on the dataset with ground truth or without ground truth
labels = pd.read_csv('xx.csv')/ np.load('xx.npy')
y_test = labels ##### Ground Truth
batches = np.load('xx.npy') ###batch information

    
    
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
metrics_dict = {

    'asw_celltype':silhouette_simple(embeddings, y_test)
    'graph_clisi': graph_clisi(adj_matrix, cell_labels)
    'Graph Connectivity': graph_connectivity(embedding, np.ravel(y_test)),
    'ARI': metrics.adjusted_rand_score(np.ravel(y_test), np.ravel(y_pred)),
    'NMI': metrics.normalized_mutual_info_score(np.ravel(y_test), np.ravel(y_pred)),
    'FMI': metrics.fowlkes_mallows_score(np.ravel(y_test), np.ravel(y_pred)),
    'Silhouette Coefficient': metrics.silhouette_score(embedding, y_pred, metric='euclidean'),
    'Calinski-Harabaz Index': metrics.calinski_harabasz_score(embedding, y_pred),
    'Davies-Bouldin Index': metrics.davies_bouldin_score(embedding, y_pred),
    'Purity': purity(y_pred, y_test),
    'AMI': metrics.adjusted_mutual_info_score(y_test, y_pred),
    'Homogeneity': metrics.homogeneity_score(y_test, y_pred),
    'Completeness': metrics.completeness_score(y_test, y_pred),
    'V-measure': metrics.v_measure_score(y_test, y_pred),
    'F-measure': F_measure(y_pred, y_test),
    'Jaccard Index': jaccard(y_pred, y_test),
    'Dice Index': Dice(y_pred, y_test)
}
df = pd.DataFrame(list(metrics_dict.items()), columns=['Metric', f'{Method}_{dataset}']
df.to_csv(f'./{Method}_{dataset}_cluster_metrics_with_GT.csv', index=False)


# Fused Dataset(batches) with ground truth
metrics_dict = {
    'asw_batch':silhouette_batch(embeddings, batches, y_test)
    'graph_ilisi':graph_ilisi(adj_matrix, batches)
    'iso_labels':get_isolated_labels(y_test, batches)
    'iso_f1': isolated_labels_f1(y_test, y_pred, batches)
    'iso_asw' : isolated_labels_asw(y_test, embeddings, batches)
    'kBET' : kBET_from_knn_matrix(adj_matrix, batches, y_test) 
    'asw_celltype':silhouette_simple(embeddings, y_test)
    'graph_clisi': graph_clisi(adj_matrix, cell_labels)
    'Graph Connectivity': graph_connectivity(embedding, np.ravel(y_test)),
    'ARI': metrics.adjusted_rand_score(np.ravel(y_test), np.ravel(y_pred)),
    'NMI': metrics.normalized_mutual_info_score(np.ravel(y_test), np.ravel(y_pred)),
    'FMI': metrics.fowlkes_mallows_score(np.ravel(y_test), np.ravel(y_pred)),
    'Silhouette Coefficient': metrics.silhouette_score(embedding, y_pred, metric='euclidean'),
    'Calinski-Harabaz Index': metrics.calinski_harabasz_score(embedding, y_pred),
    'Davies-Bouldin Index': metrics.davies_bouldin_score(embedding, y_pred),
    'Purity': purity(y_pred, y_test),
    'AMI': metrics.adjusted_mutual_info_score(y_test, y_pred),
    'Homogeneity': metrics.homogeneity_score(y_test, y_pred),
    'Completeness': metrics.completeness_score(y_test, y_pred),
    'V-measure': metrics.v_measure_score(y_test, y_pred),
    'F-measure': F_measure(y_pred, y_test),
    'Jaccard Index': jaccard(y_pred, y_test),
    'Dice Index': Dice(y_pred, y_test)
}
df1 = pd.DataFrame(list(metrics_dict.items()), columns=['Metric', f'{Method}_{dataset}']
df1.to_csv(f'./{Method}_{dataset}_fusion_cluster_metrics_with_GT.csv', index=False)
                  
                  
# Dataset without ground truth
metrics_dict = {

    'Graph Connectivity': graph_connectivity(embedding, np.ravel(y_test)),
    'Silhouette Coefficient': metrics.silhouette_score(embedding, y_pred, metric='euclidean'),
    'Calinski-Harabaz Index': metrics.calinski_harabasz_score(embedding, y_pred),
    'Davies-Bouldin Index': metrics.davies_bouldin_score(embedding, y_pred),

}
df2 = pd.DataFrame(list(metrics_dict.items()), columns=['Metric', f'{Method}_{dataset}']
df2.to_csv(f'./{Method}_{dataset}_cluster_metrics_wo_GT.csv', index=False)
