import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from model_learning.algorithms.max_entropy import THETA_STR
from model_learning.util.plot import format_and_save_plot
from model_learning.util.io import get_file_changed_extension

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


def cluster_linear_rewards(results, linkage, dist_threshold, stds=1.):
    """
    Performs hierarchical agglomerative clustering of a group of linear reward functions found through IRL.
    :param list[ModelLearningResult] results: a list of linear IRL results used for clustering.
    :param str linkage: the linkage criterion of clustering algorithm.
    :param float dist_threshold: the distance above which the clusters are not joined (determines final number of clusters).
    :param float stds: the number of standard deviations above the gradient mean used for automatic cluster detection.
    :rtype: (AgglomerativeClustering, np.ndarray)
    :return: a tuple containing the agglomerative clustering algorithm fit to the reward weight vectors, and the
    an array containing all the reward weight vectors).
    """
    # performs clustering of all reward weights
    thetas = np.array([result.stats[THETA_STR] for result in results])
    clustering = AgglomerativeClustering(
        n_clusters=None, linkage=linkage, distance_threshold=dist_threshold)
    clustering.fit(thetas)

    # performs edge/slope detection
    grad = np.gradient(clustering.distances_)
    edges = np.where(grad > (grad.mean() + stds * grad.std()))[0]

    # manually update clusters if distance threshold is lower than expected
    if len(edges) > 0 and clustering.distances_[edges[0]] < dist_threshold:
        clustering.labels_ = np.full(clustering.labels_.shape, -1, dtype=int)
        clustering.distance_threshold = clustering.distances_[edges[0]]
        clustering.n_clusters_ = int(len(clustering.labels_) - edges[0])
        _update_clusters(clustering, 0, len(clustering.children_) - 1)
        clustering.labels_ = np.max(clustering.labels_) - clustering.labels_  # invert labels to follow natural order

    return clustering, thetas


def _update_clusters(clustering, cur_cluster, cur_node):
    children = clustering.children_[cur_node]
    dist = clustering.distances_[cur_node]
    for child in children:
        cur_cluster = cur_cluster if dist < clustering.distance_threshold else np.max(clustering.labels_) + 1
        if child < clustering.n_leaves_:
            clustering.labels_[child] = cur_cluster
        else:
            _update_clusters(clustering, cur_cluster, child - clustering.n_leaves_)


def get_clusters_means(clustering, thetas):
    """
    Get clusters' mean weight vectors and standard deviations.
    :param AgglomerativeClustering clustering: the clustering algorithm with the resulting labels/indexes.
    :param np.ndarray thetas: an array containing the reward weight vectors, of shape (num_points, rwd_vector_size).
    :rtype: (dict[str, list[int]], dict[str, np.ndarray])
    :return: a tuple with a dictionary containing the list of indexes of datapoints in each cluster and a dictionary
    containing an array of shape (2, rwd_vector_size) containing the mean and std_dev of the reward vector for each
    cluster.
    """
    # gets clusters
    clusters = {}
    for idx, cluster in enumerate(clustering.labels_):
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(idx)

    # mean weights within each cluster
    cluster_weights = {}
    for cluster in sorted(clusters):
        idxs = clusters[cluster]
        cluster_weights[cluster] = np.array([np.mean(thetas[idxs], axis=0), np.std(thetas[idxs], axis=0)])

    return clusters, cluster_weights


def save_mean_cluster_weights(cluster_weights, file_path, rwd_feat_names):
    """
    Saves the clusters' mean reward vectors to a CSV file.
    :param cluster_weights: a dictionary containing an array of shape (2, `rwd_vector_size`) containing the mean and
    std_dev of the reward vector for each cluster.
    :param str file_path: the path to the CSV file in which to save the reward vector means.
    :param list[str] rwd_feat_names: the names of each reward feature, of length `rwd_vector_size`.
    :return:
    """
    # file with cluster weights
    with open(file_path, 'w') as f:
        write = csv.writer(f)
        write.writerow(['Cluster'] + rwd_feat_names)
        for cluster in sorted(cluster_weights):
            write.writerow([cluster] + cluster_weights[cluster][0].tolist())


def save_clusters_info(clustering, extra_info, thetas, file_path, rwd_feat_names):
    """
    Saves the clusters' datapoint information, including extra information about each datapoint.
    :param AgglomerativeClustering clustering: the clustering algorithm with the resulting labels/indexes.
    :param dict[str, list] extra_info: a dictionary containing extra information about the datapoints, where keys are
    the labels for the information and the values are lists containing the info for each datapoint.
    :param np.ndarray thetas: an array containing the reward weight vectors, of shape (num_points, rwd_vector_size).
    :param str file_path: the path to the CSV file in which to save the clusters' info.
    :param list[str] rwd_feat_names: the names of each reward feature, of length `rwd_vector_size`.
    :return:
    """
    # file with cluster contents
    with open(file_path, 'w') as f:
        write = csv.writer(f)
        write.writerow(['Cluster'] + list(extra_info.keys()) + rwd_feat_names)
        write.writerows(list(zip(clustering.labels_, *list(extra_info.values()), *thetas.T.tolist())))


def plot_clustering_distances(clustering, file_path):
    """
    Saves a plot with the clustering distances resulting from the given clustering algorithm.
    :param AgglomerativeClustering clustering: the clustering algorithm with the resulting distances.
    :param str file_path: the path to the file in which to save the plot.
    :return:
    """
    # saves csv with distances
    num_clusters = np.flip(np.arange(len(clustering.distances_) + 1) + 1)
    distances = np.hstack(([0], clustering.distances_))
    np.savetxt(get_file_changed_extension(file_path, 'csv'), np.column_stack((num_clusters, distances)), '%s', ',',
               header='Num. Clusters,Distance', comments='')

    # plots distances
    plt.figure()
    plt.plot(num_clusters, distances)
    plt.xlim(num_clusters[0], num_clusters[-1])  # invert for more natural view of hierarchical clustering
    plt.ylim(ymin=0)
    plt.axvline(x=clustering.n_clusters_, c='red', ls='--', lw=clustering.distance_threshold)
    format_and_save_plot(plt.gca(), 'Reward Weights Clustering Distance', file_path,
                         x_label='Num. Clusters', show_legend=False)


def plot_clustering_dendrogram(clustering, file_path, labels=None):
    """
    Saves a dendrogram plot with the clustering resulting from the given model.
    :param AgglomerativeClustering clustering: the clustering algorithm with the resulting labels and distances.
    :param str file_path: the path to the file in which to save the plot.
    :param list[str] labels: a list containing a label for each clustering datapoint. If `None`, the cluster if of each
    datapoint is used as label.
    :return:
    """
    # saves linkage info to csv
    linkage_matrix = get_linkage_matrix(clustering)
    np.savetxt(get_file_changed_extension(file_path, 'csv'), linkage_matrix, '%s', ',',
               header='Child 0, Child 1, Distance, Leaf Count', comments='')

    # saves dendrogram plot
    labels = [str(c) for c in clustering.labels_] if labels is None else labels
    dendrogram(linkage_matrix, clustering.n_clusters_, 'level', clustering.distance_threshold,
               labels=labels, leaf_rotation=45 if max(len(l) for l in labels) > 8 else 0, leaf_font_size=8)
    plt.axhline(y=clustering.distance_threshold, c='red', ls='--', lw=clustering.distance_threshold)
    format_and_save_plot(plt.gca(), 'Reward Weights Clustering Dendrogram', file_path, show_legend=False)


def get_linkage_matrix(clustering):
    """
    Gets a linkage matrix from the `sklearn` clustering model.
    See: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    :param AgglomerativeClustering clustering: the clustering model.
    :return:
    """
    # create the counts of samples under each node
    counts = np.zeros(clustering.children_.shape[0])
    n_samples = len(clustering.labels_)
    for i, merge in enumerate(clustering.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    return np.column_stack([clustering.children_, clustering.distances_, counts]).astype(float)
