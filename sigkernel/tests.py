import numpy as np
import matplotlib.pyplot as plt

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