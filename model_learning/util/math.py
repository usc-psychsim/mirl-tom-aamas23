from typing import Tuple, Callable, Dict, Union, Optional, Literal, get_args
import numpy as np
import scipy.stats as stats
import scipy.spatial.distance as distance

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


def variation_ratio(dist: np.ndarray) -> float:
    """
    Gets the statistical dispersion of a given nominal distribution in [0,1]. The larger the ratio (-> 1), the more
    differentiated or dispersed the data are. The smaller the ratio (0 <-), the more concentrated the distribution is.
    See: https://en.wikipedia.org/wiki/Variation_ratio
    :param np.ndarray dist: the nominal distribution.
    :rtype: float
    :return: a measure of the dispersion of the data in [0,1].
    """
    total = np.sum(dist)
    mode = np.max(dist)
    return 1. - mode / total


def evenness_index(dist: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Computes Pielou's evenness index of a given nominal distribution, corresponding to the normalized Shannon diversity
    or entropy. It has into account the number of different categories one would expect to find in the distribution,
    i.e. it handles 0 entries.
    See: https://en.wikipedia.org/wiki/Species_evenness
    See: https://en.wikipedia.org/wiki/Diversity_index#Shannon_index
    References:
    Pielou, E. C. (1966). The measurement of diversity in different types of biological collections. Journal of
    theoretical biology, 13, 131-144.
    :param np.ndarray dist: an array containing the nominal distribution(s).
    :param int axis: the axis along which to compute the evenness index.
    :rtype: np.ndarray
    :return: a measure of the evenness in the [0,1] interval for each of the given nominal distributions,
    corresponding to an array of the same shape as the input but where the `axis` dimension is collapsed.
    The larger the index (-> 1), the more differentiated or even the data are.
    The smaller the index (-> 0), the more concentrated the distribution is, i.e., the more uneven the data are.
    """
    dist = np.asarray(dist)
    assert (axis < 0 and -axis <= len(dist.shape)) or (0 <= axis < len(dist.shape)), \
        f'Axis {axis} is out of bounds for array of dimension {len(dist.shape)}'
    dist = np.divide(dist, np.sum(dist, axis=axis, keepdims=True))  # normalize sum to ensure prob. distribution
    num_expected_elems = dist.shape[axis]
    with np.errstate(divide='ignore'):
        log = np.log(dist)
        log[np.isinf(log)] = 0
        entropy = - np.sum(dist * log, axis=axis)
    return entropy / np.log(num_expected_elems)


def ordinal_dispersion(dist: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Computes Leik's dispersion index of an ordinal categorical distribution. It uses the cumulative frequency
    distribution to determine ordinal dispersion.
    See: https://en.wikipedia.org/wiki/Qualitative_variation#Leik's_D
    See: https://rdrr.io/rforge/agrmt/man/Leik.html
    References:
    Leik, R. K. (1966). A measure of ordinal consensus. Pacific Sociological Review, 9(2), 85-90.
    Note: implementation based on the Leik function of the agrmt R-package:
    https://r-forge.r-project.org/scm/viewvc.php/pkg/R/Leik.R?view=markup&root=agrmt
    :param np.ndarray dist: an array containing the nominal distribution(s), assumed to be ordinal with respect to the
    indices.
    :param int axis: the axis along which to compute the dispersion index.
    :rtype: np.ndarray
    :return: a measure of the dispersion in the [0,1] interval for each of the given ordinal data distributions,
    corresponding to an array of the same shape as the input but where the `axis` dimension is collapsed.
    For each distribution, if all observations are in the same category, ordinal dispersion is 0 (unimodality/
    agreement). With half the observations in one extreme category, and half the observations in the other extreme,
    Leik's measure gives a value of 1 (bimodality/polarization). The mid-point, denoting high dispersion/uniformity,
    depends on the number of categories, tending toward 0.5 as the number of categories increases.
    """
    dist = np.asarray(dist)
    assert axis == -1 or axis < len(dist.shape), \
        f'Axis {axis} is out of bounds for array of dimension {len(dist.shape)}'

    dist = np.divide(dist, np.sum(dist, axis=axis, keepdims=True))  # normalize sum to ensure prob. dist.
    m = dist.shape[axis]  # number of categories
    r = np.cumsum(dist, axis=axis)

    # def SDi <- sapply(1:m, function(x) ifelse(r[x] <= .5, r[x], 1-r[x]))  # Differences, eqn.1
    sdi = np.empty_like(r)
    comp = r <= 0.5
    d = np.where(comp)
    not_d = np.where(np.logical_not(comp))
    sdi[d] = r[d]
    sdi[not_d] = 1. - r[not_d]
    max_sdi = .5 * (m - 1)  # Maximum possible SDi given n, eqn.2
    return np.sum(sdi, axis=axis) / max_sdi  # ordinal dispersion; standardized, eqn.3


def gaussian_variation_coefficient(mu: np.ndarray, sigma: np.ndarray, diagonal: bool = True, clip: bool = True) -> \
        np.ndarray:
    """
    Gets the coefficient of variation (CV) for the Gaussian distribution(s) parameterized by the given mean and
    covariance arrays. This corresponds to the coefficient by Reyment, 1960 which supports extension to the
    multivariate case.
    See: https://en.wikipedia.org/wiki/Coefficient_of_variation
    References:
    Reyment, R.A. (1960). Studies on Nigerian upper cretaceous and lower tertiary ostracoda: Part 1. Senonian and
    maastrichtian ostracoda. Stockolm Contributions in Geology, 7, 1–238.
    :param np.ndarray mu: an array representing the mean parameter of the Gaussian distribution(s).
    :param np.ndarray sigma: an array representing the (co)variance parameter of the Gaussian distribution(s).
    If `diagonal` is True, it is assumed that the Gaussian distribution(s) have diagonal covariance (variables are
    independent) and that the last dimension of this array represents the variances' vectors. If `diagonal` is False,
    then it is assumed that the last two dimensions of this vector represent the distribution(s)' covariance matrices.
    :param bool diagonal: whether `sigma` represents parameters of a diagonal Gaussian, i.e., where the covariance
    matrix is 0 everywhere except in the diagonal, meaning the variables are independent.
    :param bool clip: whether to clip the resulting coefficients in the [0,1] interval such that they can express
    percentages.
    :rtype: np.ndarray
    :return: an array containing the entropy of the Gaussian distribution(s), with the same shape of `mean`, but where
    the last dimension has been collapsed.
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    assert (diagonal and mu.shape == sigma.shape) or (not diagonal and mu.shape == sigma.shape[:-1]), \
        f'Incompatible mean and covariance shapes: {mu.shape}, {sigma.shape}, with diagonal: {diagonal}.'
    p = mu.shape[-1]
    det = np.prod(sigma, axis=-1) if diagonal else np.linalg.det(sigma)
    scale = np.linalg.norm(mu, axis=-1) ** 2  # extension to multivariate, multi-dimensional case
    with np.errstate(divide='ignore'):
        cv = np.sqrt(np.power(det, 1 / p) / scale)
        return np.clip(cv, 0., 1.) if clip else cv


def gaussian_entropy_dispersion(mean: np.ndarray, std: np.ndarray, clip: bool = True) -> np.ndarray:
    """
    Computes the entropy-based dispersion coefficient for univariate Gaussian distributions proposed by Kostal and
    Marsalek.
    References:
    Kostal, L., & Marsalek, P. (2010). Neuronal jitter: can we measure the spike timing dispersion differently.
    Chinese Journal of Physiology 53(6): 454-464.
    :param np.ndarray mean: the distribution(s) means.
    :param np.ndarray std: the distributions(s) standard deviations.
    :param bool clip: whether to clip the resulting coefficients in the [0,1] interval such that they can express
    percentages.
    :rtype: np.ndarray
    :return: the entropy-based dispersion coefficient computed element-wise for the given Gaussian parameters,
    resulting in an array of the same shape as `mean`.
    """
    mean = np.asarray(mean)
    std = np.asarray(std)
    assert mean.shape == std.shape, f'Means and standard deviations need to be equally shaped, but arrays of shapes' \
                                    f'{mean.shape} and {std.shape} given.'
    entropy = gaussian_entropy(std ** 2, multivariate=False)
    disp = np.exp(entropy - 1)
    disp = disp / np.abs(mean)
    return np.clip(disp, 0, 1) if clip else disp


def outliers_double_mads(data: np.ndarray, thresh: float = 3.5) -> np.ndarray:
    """
    Identifies outliers in a given data set according to the data-points' "median absolute deviation" (MAD), i.e.,
    measures the distance of all points from the median in terms of median distance.
    From answer at: https://stackoverflow.com/a/29222992
    :param np.ndarray data: the data from which to extract the outliers.
    :param float thresh: the z-score threshold above which a data-point is considered an outlier.
    :rtype: np.ndarray
    :return: an array containing the indexes of the data that are considered outliers.
    """
    # warning: this function does not check for NAs nor does it address
    # issues when more than 50% of your data have identical values
    m = np.median(data)
    abs_dev = np.abs(data - m)
    left_mad = np.median(abs_dev[data <= m])
    right_mad = np.median(abs_dev[data >= m])
    y_mad = left_mad * np.ones(len(data))
    y_mad[data > m] = right_mad
    modified_z_score = 0.6745 * abs_dev / y_mad
    modified_z_score[data == m] = 0
    return np.where(modified_z_score > thresh)[0]


def outliers_dist_mean(data: np.ndarray, std_devs: float = 2.,
                       above: bool = True, below: bool = True) -> np.ndarray:
    """
    Identifies outliers according to distance of a number of standard deviations to the mean.
    :param np.ndarray data: the data from which to extract the outliers.
    :param float std_devs: the number of standard deviations above/below which a point is considered an outlier.
    :param bool above: whether to consider outliers above the mean.
    :param bool below: whether to consider outliers below the mean.
    :rtype: np.ndarray
    :return: an array containing the indexes of the data that are considered outliers.
    """
    mean = np.mean(data)
    std = np.std(data)
    outliers = [False] * len(data)
    if above:
        outliers |= data >= mean + std_devs * std
    if below:
        outliers |= data <= mean - std_devs * std
    return np.where(outliers)[0]


def pairwise_jensen_shannon_divergence(dist1: np.ndarray, dist2: np.ndarray) -> np.ndarray:
    """
    Computes the Jensen-Shannon divergence (JSD) between the two given sets of discrete probability distributions.
    Higher values (close to 1) mean that the distributions are very dissimilar while low values (close to 0) denote a
    low divergence, i.e., similar distributions.
    See: https://stackoverflow.com/a/40545237
    See: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    References:
    Lin J. 1991. "Divergence Measures Based on the Shannon Entropy". IEEE Transactions on Information Theory.
    (33) 1: 145-151.
    :param np.ndarray dist1: the array containing the first (set of) discrete probability distributions on the last dimension.
    :param np.ndarray dist2: the array containing the second (set of) discrete probability distributions on the last dimension.
    :rtype: np.ndarray
    :return: the divergence between the two sets of distributions in [0,1], where divergences are computed pairwise,
    resulting in an array with the same shape as `dist1` and `dist2` except the last dimension, which is collapsed.
    """
    dist1 = np.asarray(dist1)
    dist2 = np.asarray(dist2)
    assert dist1.shape == dist2.shape, f'Distribution shapes do not match: {dist1.shape} and {dist2.shape}.'

    def _kl_div(a, b):
        return np.nansum(a * np.log2(a / b), axis=-1)

    m = 0.5 * (dist1 + dist2)
    return 0.5 * (_kl_div(dist1, m) + _kl_div(dist2, m))


def jensen_shannon_divergence(dists: np.ndarray) -> np.ndarray:
    """
    Computes the Jensen-Shannon divergence (JSD) between the given discrete probability distributions. The JSD is
    normalized by log2(n_dims) to fall in the [0,1] interval. Higher values (close to 1) mean that the distributions
    are very dissimilar while low values (close to 0) denote a low divergence, i.e., similar distributions.
    See: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    References:
    Lin J. 1991. "Divergence Measures Based on the Shannon Entropy". IEEE Transactions on Information Theory.
    (33) 1: 145-151.
    :param np.ndarray dists: the array containing the discrete probability distributions on the last dimension.
    :rtype: np.ndarray
    :return: the divergence between the given distributions in [0,1], resulting in an array with the same shape as
    `dists` except the last two dimensions were collapsed.
    """
    dists = np.asarray(dists)
    assert len(dists.shape) >= 2, \
        f'Distribution array must be at least 2D to compute the JSD, but array of shape {dists.shape} was given.'
    n = dists.shape[-2]
    m = np.sum(dists, axis=-2, keepdims=True) / n
    dkl = np.nansum(dists * np.log2(dists / m), axis=-1)
    return np.sum(dkl, axis=-1) / (n * np.log2(n))


def decomposed_jensen_shannon_divergence(dist1: np.ndarray, dist2: np.ndarray) -> np.ndarray:
    """
    Computes the pairwise Jensen-Shannon divergence between two discrete probability distributions. This corresponds to
    the un-summed JSD, i.e., the divergence according to each component of the given distributions. Summing up the
    returned array yields the true JSD between the two distributions.
    See: https://stackoverflow.com/a/40545237
    See: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    Input must be two probability distributions of equal length that sum to 1.
    :param np.ndarray dist1: the first discrete probability distribution.
    :param np.ndarray dist2: the second discrete probability distribution.
    :rtype: np.ndarray
    :return: an array the same size of the distributions with the divergence between each component in [0,1].
    """
    assert dist1.shape == dist2.shape, 'Distribution shapes do not match'

    def _kl_div(a, b):
        return a * np.log2(a / b)

    m = 0.5 * (dist1 + dist2)
    return 0.5 * (_kl_div(dist1, m) + _kl_div(dist2, m))


def jensen_renyi_divergence(mu: np.ndarray, sigma: np.ndarray,
                            min_log_sigma: float = -5, max_log_sigma: float = -1, decay: float = 0.1,
                            clip: bool = True) -> np.ndarray:
    """
    Computes the Jensen-Rényi Divergence (JRD) using the quadratic Rényi entropy between several multivariate Gaussian
    distributions, resulting in a metric in the [0, 1] interval. Higher values (close to 1) mean that the distributions
    are very dissimilar while low values (close to 0) denote a low divergence, i.e., similar distributions.
    Based on the implementation in:
    https://github.com/nnaisense/MAX/blob/36d9ed6bc4ca108f168ddd6a50f14ce5a52a3644/utilities.py#L126
    References:
    Wang, F., Syeda-Mahmood, T., Vemuri, B. C., Beymer, D., & Rangarajan, A. (2009). Closed-form Jensen-Renyi
    divergence for mixture of Gaussians and applications to group-wise shape registration. In International Conference
    on Medical Image Computing and Computer-Assisted Intervention (pp. 648-655). Springer, Berlin, Heidelberg.
    Shyam, P., Jaśkowski, W., & Gomez, F. (2019). Model-based active exploration. In International conference on
    machine learning (pp. 5779-5788). PMLR
    :param np.ndarray mu: the mean vectors of the Gaussian distributions. It is assumed that the multivariate
    distributions are in the last dimension, while the JRD is computed between distributions across the second to last
    dimension, i.e., expected input shape: (*, n_dists, dist_size).
    :param np.ndarray sigma: the covariance diagonals of the Gaussian distributions, shape: (*, n_dists, dist_size).
    :param float min_log_sigma: the lower bound for the log of the variances.
    :param float max_log_sigma: the upper bound for the log of the variances.
    :param float decay: temperature parameter used to rescale the variance.
    :param bool clip: whether to clip the resulting JRDs in the [0,1] interval such that they can express percentages.
    :rtype: np.ndarray
    :return: an array containing the JRDs between the given distributions, resulting in an array with the same shape as
    `mu` but with the last two dimensions collapsed.
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    assert len(mu.shape) >= 2, f'Means should represent multiple multivariate Gaussian distributions, so need to be ' \
                               f'at least 2D, but array of shape {mu.shape} given.'
    assert mu.shape == sigma.shape, f'Means and variances need to be equally shaped, but arrays of shapes' \
                                    f'{mu.shape} and {sigma.shape} given.'

    n, d = mu.shape[-2], mu.shape[-1]  # input shape: (*, n_dists, dist_size)
    max_sigma = np.exp(max_log_sigma)
    sigma = max_sigma - decay * (max_sigma - sigma)

    # entropy of the mean
    mu_diff = np.expand_dims(mu, -3) - np.expand_dims(mu, -2)  # shape: (*, n_dists, n_dists, dist_size)
    var_sum = np.expand_dims(sigma, -3) + np.expand_dims(sigma, -2)  # shape: (*, n_dists, n_dists, dist_size)

    err = (mu_diff * 1 / var_sum * mu_diff)  # shape: (*, n_dists, n_dists, dist_size)
    err = np.sum(err, axis=-1)  # shape: (*, n_dists, n_dists)
    det = np.sum(np.log(var_sum), axis=-1)  # shape: (*, n_dists, n_dists)

    log_z = -0.5 * (err + det)  # shape: (*, n_dists, n_dists)
    log_z = log_z.reshape(log_z.shape[:-2] + (n * n,))  # shape: (*, n_dists * n_dists)
    mx = np.max(log_z, axis=-1, keepdims=True)  # shape: (*, 1)
    log_z = log_z - mx  # shape: (*, n_dists * n_dists)
    exp = np.mean(np.exp(log_z), axis=-1, keepdims=True)  # shape: (*, 1)
    entropy_mean = -mx - np.log(exp)  # shape: (*, 1)
    entropy_mean = entropy_mean.reshape(entropy_mean.shape[:-1])  # shape: (*)

    # mean of entropies
    total_entropy = np.sum(np.log(sigma), axis=-1)  # shape: (*, n_dists)
    mean_entropy = np.mean(total_entropy, axis=-1) / 2 + d * np.log(2) / 2  # shape: (*)

    # jensen-renyi divergence
    jrd = entropy_mean - mean_entropy  # shape: (*)
    return np.clip(jrd, 0, 1) if clip else jrd


def gaussian_diff_means(mean1: float, std1: int, n1: int, mean2: float, std2: float, n2: int) -> \
        Tuple[float, float, int]:
    """
    Gets the difference of the given sample means (mean1 - mean2).
    See: https://stattrek.com/sampling/difference-in-means.aspx
    :param float mean1: the first mean value.
    :param float std1: the first mean's standard deviation.
    :param int n1: the first mean's count.
    :param float mean2: the first mean value.
    :param float std2: the first mean's standard deviation.
    :param int n2: the first mean's count.
    :rtype: (float, float, int)
    :return: a tuple containing the differences of the mean, standard deviation and number of elements.
    """
    return \
        mean1 - mean2, \
        np.sqrt((std1 * std1) / n1 + (std2 * std2) / n2).item(), \
        n1 - n2


def gaussian_kl_divergence(mu1: np.ndarray, mu2: np.ndarray, sigma_1: np.ndarray, sigma_2: np.ndarray) -> float:
    """
    Computes the Kullback-Leibler divergence (KL-D) between one Gaussian distribution and another Gaussian distribution
    (or set of distributions). Distributions assumed to have diagonal covariances.
    See: https://stackoverflow.com/q/44549369
    :param np.ndarray mu1: the mean vector of the first distribution.
    :param np.ndarray mu2: the mean vector of the second distribution (or set of distributions).
    :param np.ndarray sigma_1: the covariance diagonal of the first distribution.
    :param np.ndarray sigma_2: the covariance diagonal of the second distribution (or set of distributions).
    :rtype: float
    :return: the KL divergence between the two distributions.
    """
    if len(mu2.shape) == 2:
        axis = 1
    else:
        axis = 0
    mu_diff = mu2 - mu1
    return 0.5 * (np.log(np.prod(sigma_2, axis=axis) / np.prod(sigma_1))
                  - mu1.shape[0] + np.sum(sigma_1 / sigma_2, axis=axis)
                  + np.sum(mu_diff * 1 / sigma_2 * mu_diff, axis=axis))


def gaussian_entropy(sigma: np.ndarray, multivariate: bool = True, diagonal: bool = True) -> np.ndarray:
    """
    Computes the entropy of a (possibly multivariate) Gaussian distribution with the given (co)variance.
    See: https://sgfin.github.io/2017/03/11/Deriving-the-information-entropy-of-the-multivariate-gaussian/
    :param np.ndarray sigma: an array representing the (co)variance parameter of the Gaussian distribution(s).
    If `multivariate` is `False`, then each element in this array corresponds to the variance of a different univariate
    Gaussian distribution. If `multivariate` is `True`, then this array parameterizes multivariate Gaussians in the last
    dimensions of the array. Namely, if `diagonal` is True, it is assumed that the Gaussian distribution(s) have
    diagonal  covariance (variables are independent) and that the last dimension of this array represents the variances'
    vectors. If `diagonal` is False, then it is assumed that the last two dimensions of this vector represent the
    distribution(s)' covariance matrices.
    :param bool multivariate: whether the given variance parameterizes multivariate (`True`) or univariate (`False`)
    Gaussian distributions.
    :param bool diagonal: whether `sigma` represents parameters of a diagonal Gaussian, i.e., where the covariance
    matrix is 0 everywhere except in the diagonal, meaning the variables are independent.
    :rtype: np.ndarray
    :return: an array containing the entropy of the Gaussian distribution. If `multivariate` is `False`, then the
    array will have the same shape of `sigma`. If `multivariate` is `True`, then the array will have the same shape as
    `sigma` except the last dimension, if `diagonal` is `True`, or last two dimensions, if `diagonal` is `False`,
    have been collapsed.
    """
    # if covariance matrix, last 2 dimensions of the array must be square
    sigma = np.asarray(sigma)
    assert not multivariate or diagonal or (len(sigma.shape) >= 2 and sigma.shape[-1] == sigma.shape[-2]), \
        f'Covariance matrix needs to be a square, but matrix of shape {sigma.shape[-2:]} provided.'
    n = sigma.shape[-1] if multivariate else 1
    det = sigma if not multivariate else np.prod(sigma, axis=-1) if diagonal else np.linalg.det(sigma)
    return n / 2 * np.log(2 * np.pi * np.e) + 0.5 * np.log(det)


def gaussian_pdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, diagonal: bool = True) -> np.ndarray:
    """
    Applies the probability density function (PDF) of multivariate Gaussian distributions to the given inputs.
    See: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    :param np.ndarray x: the inputs to get the PDF, an array shaped (*, n_dims).
    :param np.ndarray mu: the means of the Gaussian distributions, an array shaped (*, n_dims).
    :param np.ndarray sigma: the covariance matrices of the Gaussian distributions, an array shaped (*, n_dims) if
    `diagonal` is `True`, else (*, n_dims, n_dims).
    :param bool diagonal: whether `sigma` represents parameters of a diagonal Gaussian, i.e., where the covariance
    matrix is 0 everywhere except in the diagonal, meaning the variables are independent.
    :rtype: np.ndarray
    :return: the PDF of the given multivariate Gaussian distributions applied to the given inputs, resulting in an
    array of shape (`x.shape[0]`,)
    """
    x = np.asarray(x)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    assert ((diagonal and mu.shape == sigma.shape and x.shape == mu.shape) or
            (not diagonal and len(sigma.shape) >= 2 and sigma.shape[-1] == sigma.shape[-2] and
             sigma.shape[:2] == mu.shape and mu.shape == x.shape)), \
        f'Mean and covariance have incompatible dimensions: {mu.shape} and {sigma.shape}.'

    k = mu.shape[-1]
    det = np.prod(sigma, axis=-1) if diagonal else np.linalg.det(sigma)
    inv = 1 / sigma if diagonal else np.linalg.inv(sigma)
    diff = x - mu
    diff_mul = np.sum(diff * inv * diff, axis=-1) if diagonal else np.dot(np.dot(diff.T, inv), diff)
    pdf = np.power(2 * np.pi, -0.5 * k) * np.power(det, -0.5) * np.exp(-0.5 * diff_mul)
    return pdf


def discretize_gaussian(mu: Union[float, np.ndarray], sigma: Union[float, np.ndarray], n_bins: int = 100,
                        lower: Union[float, np.ndarray] = None, upper: Union[float, np.ndarray] = None,
                        coverage: Union[float, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretizes one or more Gaussian distributions into N bins according to the given parameters.
    :param float or np.ndarray mu: the Gaussian's mean value(s).
    :param float or np.ndarray sigma: the Gaussian standard deviation (square root of variance).
    :param int n_bins: the number of intervals for the discretization.
    :param float or np.ndarray lower: the lower bound of the interval for which to discretize.
    Default is `None`, i.e., no bound, in which case `coverage` cannot be `None`.
    :param float or np.ndarray upper: the upper bound of the interval for which to discretize.
    Default is `None`, i.e., no bound, in which case `coverage` cannot be `None`.
    :param float or np.ndarray coverage: the distribution coverage (CDF) used to compute the interval for which to
    discretize. Default is `None`, i.e., interval is given by `lower` and `upper` bounds.
    :rtype: (np.ndarray, np.ndarray)
    :return: a tuple containing the bins' center values and the corresponding probabilities after discretization, each
    an array of shape (batch_size, n_bins).
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    lower = None if lower is None else np.asarray(lower)
    upper = None if upper is None else np.asarray(upper)
    coverage = None if coverage is None else np.asarray(coverage)

    if coverage is not None:
        l, u = stats.norm.interval(coverage, loc=mu, scale=sigma)
        if lower is not None:
            l = np.maximum(l, lower)
        if upper is not None:
            u = np.minimum(u, upper)
        lower, upper = l, u

    if lower is None or upper is None:
        raise ValueError(f'Undefined upper and/or lower interval bounds: ({lower, upper})')

    bins = np.linspace(lower, upper, n_bins)  # gets sample value centers, shape: (n_bins, batch)
    probs = stats.norm.pdf(bins, loc=mu, scale=sigma)  # gets corresponding sample probabilities
    probs = probs / probs.sum(axis=0, keepdims=True)  # normalize to sum 1
    return np.moveaxis(bins, 0, -1), np.moveaxis(probs, 0, -1)  # final shapes: (batch, n_bins)


def discretize_uniform(low: Union[float, np.ndarray], high: Union[float, np.ndarray], n_bins: int = 100,
                       lower: Union[float, np.ndarray] = None, upper: Union[float, np.ndarray] = None) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretizes a uniform distribution into N bins according to the given parameters.
    :param flfloat or np.ndarrayoat low: distribution's lower range (inclusive) value.
    :param float or np.ndarray high: distribution's upper range (exclusive) value.
    :param int n_bins: the number of intervals for the discretization.
    :param float or np.ndarray lower: the lower bound of the interval for which to discretize.
    Default is `None`, i.e., use `low`.
    :param float or np.ndarray upper: the upper bound of the interval for which to discretize.
    Default is `None`, i.e., use `high`.
    :rtype: (np.ndarray, np.ndarray)
    :return: a tuple containing the bins' center values and the corresponding probabilities after discretization, each
    an array of shape (batch_size, n_bins).
    """
    low = np.asarray(low)
    high = np.asarray(high)
    lower = low if lower is None else np.asarray(lower)
    upper = high if upper is None else np.asarray(upper)

    assert np.all(high > low), f'Distribution\'s upper bound must be higher than lower bound, but {high}<={low}!'
    bins = np.linspace(lower, upper, n_bins)  # gets sample value centers, shape: (n_bins, batch)
    probs = stats.uniform.pdf(bins, loc=low, scale=high - low)  # gets corresponding sample probabilities
    probs = probs / probs.sum(axis=0, keepdims=True)  # normalize to sum 1
    return np.moveaxis(bins, 0, -1), np.moveaxis(probs, 0, -1)  # final shapes: (batch, n_bins)


FiniteDiffType = Literal['backward', 'center', 'forward']


def _fd_coefficients_non_uni(order: int, accuracy: int, x: np.ndarray, idx: int, fd_type: FiniteDiffType = 'center') \
        -> Dict[str, Union[np.ndarray, int]]:
    """
    Modified version of findiff.coefs.coefficients_non_uni to allow specifying type of finite difference.
    """
    assert order > 0, f'Order has to be greater than 0, {order} provided.'
    assert accuracy > 0 and accuracy % 2 == 0, f'Accuracy has to be a positive, even number, but {accuracy} provided.'
    assert fd_type in get_args(FiniteDiffType), \
        f'Unknown finite difference type: {fd_type}, options are: {get_args(FiniteDiffType)}'

    from findiff.coefs import _build_matrix_non_uniform, _build_rhs

    num_central = 2 * np.floor((order + 1) / 2) - 1 + accuracy

    if fd_type == 'center':
        num_side = int(num_central // 2)
        matrix = _build_matrix_non_uniform(num_side, num_side, x, idx)
        offsets = np.arange(-num_side, num_side + 1)
        rhs = _build_rhs(offsets, order)
        return dict(coefficients=np.linalg.solve(matrix, rhs), offsets=offsets, accuracy=accuracy)

    num_coef = int(num_central + 1 if order % 2 == 0 else num_central)

    if fd_type == 'backward':
        matrix = _build_matrix_non_uniform(num_coef - 1, 0, x, idx)
        offsets = np.arange(-num_coef + 1, 1)
        rhs = _build_rhs(offsets, order)
        return dict(coefficients=np.linalg.solve(matrix, rhs), offsets=offsets, accuracy=accuracy)

    # forward
    matrix = _build_matrix_non_uniform(0, num_coef - 1, x, idx)
    offsets = np.arange(num_coef)
    rhs = _build_rhs(offsets, order)
    return dict(coefficients=np.linalg.solve(matrix, rhs), offsets=offsets, accuracy=accuracy)


def _fd_coefficients_uni(order: int, accuracy: int, fd_type: FiniteDiffType = 'center') \
        -> Dict[str, Union[np.ndarray, int]]:
    """
    Copy of  findiff.coefs.coefficients to allow specifying type of finite difference.
    """
    assert order > 0, f'Order has to be greater than 0, {order} provided.'
    assert accuracy > 0 and accuracy % 2 == 0, f'Accuracy has to be a positive, even number, but {accuracy} provided.'
    assert fd_type in get_args(FiniteDiffType), \
        f'Unknown finite difference type: {fd_type}, options are: {get_args(FiniteDiffType)}'

    from findiff.coefs import calc_coefs

    num_central = 2 * np.floor((order + 1) / 2) - 1 + accuracy

    if fd_type == 'center':
        # center
        num_side = int(num_central // 2)
        offsets = np.arange(-num_side, num_side + 1)
        return calc_coefs(order, offsets, symbolic=False)

    num_coef = int(num_central + 1 if order % 2 == 0 else num_central)

    if fd_type == 'backward':
        offsets = np.arange(-num_coef + 1, 1)
        return calc_coefs(order, offsets, symbolic=False)

    # forward
    offsets = np.arange(num_coef)
    return calc_coefs(order, offsets, symbolic=False)


def finite_diff(y: np.ndarray,
                x: Optional[Union[np.ndarray, float]] = 1.,
                order: int = 1,
                accuracy: int = 2,
                fd_type: FiniteDiffType = 'center',
                default_value: float = np.nan) -> np.ndarray:
    """
    Computes the finite difference numerical derivative of a function based on the given samples.
    See: https://en.wikipedia.org/wiki/Finite_difference
    :param np.ndarray y: the input array with the function sample values to be derived.
    :param np.ndarray x: the x-coordinates corresponding to each y value, or the uniform spacing value.
    If `None`, it is assumed that the data is on a uniform grid of size 1.
    :param int order: the order of the derivative to be computed, default is 1.
    :param int accuracy: the finite difference accuracy (has to be a positive even number), default is 2.
    :param str fd_type: the type of finite difference to compute which dictates where to sample points to estimate
    the derivatives.
    :param float default_value: the default value to be returned at points where the derivative cannot be estimated.
    :rtype: np.ndarray
    :return: an array with the same shape as the input array containing the derivative's values.
    """
    assert x is None or not isinstance(x, np.ndarray) or x.shape == y.shape, \
        f'Shapes of `x` and `y` have to be the same, but got {x.shape} and {y.shape}, respectively.'
    assert order > 0, f'Order has to be greater than 0, {order} provided.'
    assert accuracy > 0 and accuracy % 2 == 0, f'Accuracy has to be a positive, even number, but {accuracy} provided.'
    assert fd_type in get_args(FiniteDiffType), \
        f'Unknown finite difference type: {fd_type}, options are: {get_args(FiniteDiffType)}'

    if x is None:
        x = 1.

    num_central = 2 * np.floor((order + 1) / 2) - 1 + accuracy
    num_coef = num_central + 1 if order % 2 == 0 else num_central
    num_side = num_central // 2

    def _insuf_data(idx):
        return ((fd_type == 'backward' and idx < num_coef - 1) or
                (fd_type == 'center' and (idx < num_side or idx >= len(y) - num_side)) or
                (fd_type == 'forward' and idx > len(y) - num_coef))

    # check uniformity of space
    dy_dx = np.full_like(y, default_value, dtype=np.float64)
    if isinstance(x, np.ndarray):
        # non-uniform grid, diff coefficients for each index
        for idx in range(len(y)):
            if _insuf_data(idx):
                continue  # ignore, not enough data
            coefs = _fd_coefficients_non_uni(order, accuracy, x, idx, fd_type)
            dy_dx[idx] = np.dot(y[idx + coefs['offsets']], coefs['coefficients'])
    else:
        # uniform grid space, same coefficients for all indices
        coefs = _fd_coefficients_uni(order, accuracy, fd_type)
        offsets = coefs['offsets']
        coefs = float(x) ** -order * coefs['coefficients']
        for idx in range(len(y)):
            if _insuf_data(idx):
                continue  # ignore, not enough data
            dy_dx[idx] = np.dot(y[idx + offsets], coefs)

    return dy_dx


def normalized_derivative(y: np.ndarray,
                          x: Optional[Union[np.ndarray, float]] = 1.,
                          order: int = 1,
                          accuracy: int = 2,
                          fd_type: FiniteDiffType = 'center',
                          default_value: float = np.nan) -> np.ndarray:
    """
    Computes the n-th derivative of the given array using a finite difference numerical method, and then normalizes
    the derivative using the sine function (of the angle whose tangent is given by the derivative).
    :param np.ndarray y: the input array to be normalized.
    :param np.ndarray x: the x-coordinates corresponding to each y value, or the uniform spacing value.
    If `None`, it is assumed that the data is on a uniform grid of size 1.
    :param int order: the order of the derivative to be computed, default is 1.
    :param int accuracy: the finite difference accuracy (has to be a positive even number), default is 2.
    :param str fd_type: the type of finite difference to compute which dictates where to sample points to estimate
    the derivatives.
    :param float default_value: the default value to be returned at points where the derivative cannot be estimated.
    :rtype: np.ndarray
    :return: an array with the same shape as the input array containing the derivative's sine-normalized values.
    """
    assert x is None or not isinstance(x, np.ndarray) or x.shape == y.shape, \
        f'Shapes of `x` and `y` have to be the same, but got {x.shape} and {y.shape}, respectively.'
    assert order > 0, f'Order has to be greater than 0, {order} provided.'
    assert accuracy > 0 and accuracy % 2 == 0, f'Accuracy has to be a positive, even number, but {accuracy} provided.'

    # gets finite difference of given order and accuracy
    dy_dx = finite_diff(y, x, order, accuracy, fd_type, default_value)

    # computes angles and get sine
    angles = np.arctan(dy_dx)
    norm = np.sin(angles)
    return norm


def apply_numpy_function(a: np.ndarray, func: Callable[[np.ndarray], np.ndarray], axis: int) -> np.ndarray:
    """
    Applies a function over all elements of the given numpy array up to a certain axis, while the rest of the data is
    used as the argument for the given function.
    Based on: https://codereview.stackexchange.com/a/90404
    :param np.ndarray a: the input array containing the data.
    :param func: the function to be called element-wise, has to receive an array as input and outputs an array.
    :param int axis: the axis that defines the shape of the arrays being input to the function `func`, i.e., the input
    arrays to the function will have shape `arr.shape[axis:]`.
    :rtype: np.ndarray
    :return: an array with the results of applying the given function element-wise to the given array up to the
    specified axis, resulting in an array of shape `arr.shape[:axis] + outputs.shape`, where outputs of the function
    should have the same shape.
    """
    a = np.asarray(a)
    if axis < 0:
        axis = len(a.shape) + axis
    assert 0 <= axis < len(a.shape), f'Axis {axis} out-of-bounds for array of shape {a.shape}.'
    grid = [np.arange(a.shape[i]) for i in range(axis)]
    grid = np.meshgrid(*grid)
    vec = np.vectorize(func, otypes=[np.ndarray])
    res = vec(*grid)
    return np.array(res.tolist())


def mean_pairwise_distances(a: np.ndarray, metric: str = 'euclidean', **kwargs: Dict):
    """
    Computes the mean pairwise distances between the given observations using Scipy's `pdist` function.
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
    :param np.ndarray a: the input array containing the n-dimensional observations, shaped: (*, n_obs, n_dims).
    :param str metric: the name of the Scipy distance metric to be used.
    :param dict kwargs: extra arguments to be passed to the Scipy's distance function.
    :rtype: np.ndarray
    :return: an array containing the mean pairwise observation distances, with the same shape as the given array,
    except the last two dimensions which are collapsed.
    """

    def _mean_dist(*idxs: np.ndarray):
        # original shape: (*, n_obs, n_dims)
        return np.mean(distance.pdist(a[idxs], metric=metric, **kwargs), axis=-1)  # shape: (*,)

    return apply_numpy_function(a, _mean_dist, axis=-2)


def stretch_array(a: np.ndarray, length: int = 1, axis: int = 0) -> np.ndarray:
    """
    Stretches or compresses an array to a new length along some dimension.
    :param np.ndarray a: the input array to be stretched.
    :param int length: the new length of the array along the given dimension.
    :param int axis: the axis along which to stretch.
    :rtype: np.ndarray
    :return: a new array stretched (or compressed) to the given length along the given axis. If length is greater than
    the existing axis size then the values are repeated to fill in the missing values, otherwise they are sub-sampled to
    to fit the new length.
    """
    a = np.asarray(a)
    if axis < 0:
        axis = len(a.shape) + axis
    assert 0 <= axis < len(a.shape), f'Axis {axis} out-of-bounds for array of shape {a.shape}.'
    assert length > 0, f'Invalid length provided: {length}, must be positive number.'

    if length == a.shape[axis]:
        return a  # nothing to do, already correct length

    idxs = np.round(np.linspace(0, a.shape[axis] - 1, length)).astype(int)
    return a.take(indices=idxs, axis=axis)  # sub/re-sample at +- regular intervals
