import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from .kernels import SignatureKernel
from .static.kernels import LinearKernel
from .utils import matrix_diag

def two_sample_permutation_test(test_statistic, X, Y, num_permutations):
    assert X.ndim == Y.ndim

    statistics = np.zeros(num_permutations)

    for i in range(num_permutations):
        # concatenate samples
        if X.ndim == 1:
            Z = np.hstack((X,Y))
        elif X.ndim > 1:
            Z = np.vstack((X,Y))
        # permute samples and compute test statistic
        perm_inds = np.random.permutation(len(Z))
        Z = Z[perm_inds]
        X_ = Z[:len(X)]
        Y_ = Z[len(X):]
        my_test_statistic = test_statistic(X_, Y_)
        statistics[i] = my_test_statistic
    return statistics

def quadratic_time_mmd(X,Y,kernel):
    # assert X.ndim == Y.ndim == 2
    K_XX = kernel(X,X)
    K_XY = kernel(X,Y)
    K_YY = kernel(Y,Y)

    n = len(K_XX)
    m = len(K_YY)

    # unbiased MMD statistic (could also use biased, doesn't matter if we use permutation tests)
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    mmd = np.sum(K_XX) / (n*(n-1))  + np.sum(K_YY) / (m*(m-1))  - 2*np.sum(K_XY)/(n*m)
    return mmd

def psi(x, a=1, C=4):
    x = C+C**(1+a)*(C**-a - x**-a)/a if x>4 else x
    return x

def norm_func(λ, norms, norm_sum, a=1, c=4):
    m = len(norms)
    λ = np.ones(m) * λ
    powers = np.arange(m) * 2
    return np.sum(norms * np.power(λ, powers)) - psi(norm_sum, a, c)

def get_normalisation_constants(normsq_levels, a=1, C=4):
    n_samples = normsq_levels.shape[0]
    normsq = np.sum(normsq_levels, axis=1)
    norm_condition = normsq > C
    λ = np.ones(n_samples)
    for i in range(n_samples):
        if norm_condition[i]:
            λ[i] = brentq(norm_func, 0, 1, args=(np.asnumpy(normsq_levels[i]), np.asnumpy(normsq[i]).sum(), 1, C))
    return λ

def sum_normalise_gram_matrix(K, λ_X, λ_Y, n_levels):
    m_λ = (λ_X[:,np.newaxis] @ λ_Y[np.newaxis,:]) ** np.arange(n_levels+1)[:, np.newaxis, np.newaxis]
    K = np.sum(m_λ * K, axis=0)
    return K

def sig_kernel_test(X, Y, n_levels, static_kernel=LinearKernel, quantile=0.95, num_permutations=1000, a=1, C=4):
    '''
    X: np.array of shape (n_samples, n_features)
    Y: np.array of shape (n_samples, n_features)
    n_levels: int, number of levels of the signature kernel where level 0 is not included e.g. 3 means signature (t0, t1, t2, t3)
    static_kernel: static kernel to be lifted to sequence kernel e.g. LinearKernel or RBFKernel
    quantile: float, quantile to be used for the test
    num_permutations: int, number of permutations to be used to generate the null distribution
    a: float, parameter for psi function used in the tensor normalisation to get the characteristic signature kernel
    C: float, as above
    '''
    assert X.ndim == Y.ndim

    kernel = SignatureKernel(n_levels=n_levels, order=n_levels, normalization=3, static_kernel=static_kernel)

    # calculate Gram matrices
    K_XX = kernel(X,X)
    K_XY = kernel(X,Y)
    K_YY = kernel(Y,Y)

    # calculate tensor norms and which tensors need to be normalised
    normsq_levels_X = matrix_diag(K_XX).T
    normsq_levels_Y = matrix_diag(K_YY).T

    # find normalisation constants
    λ_X = get_normalisation_constants(normsq_levels_X, a, C)
    λ_Y = get_normalisation_constants(normsq_levels_Y, a, C)

    # normalise at each signature level then sum
    K_XX = sum_normalise_gram_matrix(K_XX, λ_X, λ_X, n_levels)
    K_XY = sum_normalise_gram_matrix(K_XY, λ_X, λ_Y, n_levels)
    K_YY = sum_normalise_gram_matrix(K_YY, λ_Y, λ_Y, n_levels)

    # unbiased MMD statistic (could also use biased, doesn't matter if we use permutation tests)
    n = len(K_XX)
    m = len(K_YY)
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    mmd = np.sum(K_XX) / (n*(n-1))  + np.sum(K_YY) / (m*(m-1))  - 2*np.sum(K_XY)/(n*m)

    null_mmds = np.zeros(num_permutations)
    for i in range(num_permutations):
        # permute samples and compute test statistic
        perm_inds = np.random.permutation(m+n)
        null_mmds[i] = my_test_statistic
    return mmd, null_mmds

def quadratic_time_mmd(X,Y,kernel):
    # assert X.ndim == Y.ndim == 2
    K_XX = kernel(X,X)
    K_XY = kernel(X,Y)
    K_YY = kernel(Y,Y)

    n = len(K_XX)
    m = len(K_YY)

    # unbiased MMD statistic (could also use biased, doesn't matter if we use permutation tests)
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    mmd = np.sum(K_XX) / (n*(n-1))  + np.sum(K_YY) / (m*(m-1))  - 2*np.sum(K_XY)/(n*m)
    return mmd

def plot_permutation_samples(null_samples, statistic=None):
    plt.hist(null_samples)
    plt.axvline(x=np.percentile(null_samples, 2.5), c='b')
    legend = ["95% quantiles"]
    if statistic is not None:
        plt.axvline(x=statistic, c='r')
        legend += ["Actual test statistic"]
    plt.legend(legend)
    plt.axvline(x=np.percentile(null_samples, 97.5), c='b')
    plt.xlabel("Test statistic value")
    plt.ylabel("Counts")

# simulate geometric Brownian motion paths
def gen_GBM_path(mu, sigma, dt, n_paths, seq_len):
    n_steps = seq_len - 1
    path = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(n_paths, n_steps))
    path = np.cumprod(path, axis=1)
    path = np.concatenate([np.ones((n_paths, 1)), path], axis=1)
    path = path[..., np.newaxis]
    return path # shape (n_paths, seq_len, 1)