import os
from itertools import combinations
from sklearn import metrics
from model_learning.util.plot import plot_bar

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

GT_EVAL_METRICS = {'Adjusted Rand Index': ('ari', metrics.adjusted_rand_score),
                   'Adjusted Mutual Information': ('ami', metrics.adjusted_mutual_info_score),
                   'Normalized Mutual Information': ('nmi', metrics.normalized_mutual_info_score),
                   'Fowlkes-Mallows Index': ('fmi', metrics.fowlkes_mallows_score)}


def _get_unique_labels(labels):
    unique_labels = list(set(labels))
    return [unique_labels.index(label) for label in labels]


def evaluate_clustering(clustering, gt_labels, output_dir, img_format='pdf', max_comb_length=1):
    """
    Performs clustering evaluation according to different metrics for the given data partitions.
    Also performs clustering evaluation for combinations of the ground-truth partitions.
    Prints different bar plots and CSV files for the different metrics and partition evaluations.
    :param AgglomerativeClustering clustering: the clustering algorithm with the resulting labels/indexes.
    :param dict[str, list] gt_labels: the different ground-truth data partitions, a dict in the form `part name`->`raw labels`.
    :param str output_dir: the directory in which to save the results.
    :param str img_format: the format of plot images to be saved.
    :param int max_comb_length: the maximum length of the ground-truth partition combinations to be evaluated.
    A value of `1` means that only the original, non-combined partitions, are evaluated.
    :rtype: dict[int, dict[str, dict[str, float]]]
    :return: a dictionary of form comb_length -> metric -> partition -> value  containing, for each combination length
    and for each metric, a dictionary with that metric's evaluation for a particular data partition with that length.
    """
    # gets unique labels
    gt_labels = {name: _get_unique_labels(gt_labels[name]) for name in gt_labels}

    # gets different combinations of ground-truth labels
    evaluations = {}
    data_length = len(gt_labels[next(iter(gt_labels))])
    for comb_length in range(1, max_comb_length + 1):
        gt_name_combs = list(combinations(list(gt_labels.keys()), comb_length))
        gt_comb_labels = {'Ground-truth': clustering.labels_}
        for gt_name_comb in gt_name_combs:
            comb_name = ' x '.join(gt_name_comb)
            comb_labels = [''.join([str(gt_labels[name][i]) for name in gt_name_comb]) for i in range(data_length)]
            gt_comb_labels[comb_name] = _get_unique_labels(comb_labels)

        # evaluates partitions according to different metrics
        evaluations[comb_length] = {}
        for metric_name in GT_EVAL_METRICS:
            metric_short, metric_func = GT_EVAL_METRICS[metric_name]

            clustering_evals = {name: metric_func(labels, clustering.labels_)
                                for name, labels in gt_comb_labels.items()}

            evaluations[comb_length][metric_name] = clustering_evals

            plot_bar(clustering_evals, 'Clustering Evaluations',
                     os.path.join(output_dir, 'eval-comb-{}-{}.{}'.format(comb_length, metric_short, img_format)),
                     y_label=metric_name)

    return evaluations
