import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import QuantLib as ql

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
    # helper function to calculate the normalisation constant for the characteristic signature kernel
    x = C+C**(1+a)*(C**-a - x**-a)/a if x>4 else x
    return x

def norm_func(λ, norms, a=1, c=4):
    # function to solve for root which are the normalisation constants for the characteristic signature kernel
    norm_sum = norms.sum()
    m = len(norms)
    λ = np.ones(m) * λ
    powers = np.arange(m) * 2
    return np.sum(norms * np.power(λ, powers)) - psi(norm_sum, a, c)

def get_normalisation_constants(gram_matrix, a=1, C=4):
    '''
    Calculate normalisation constants for each path
    normsq_levels: np.array of shape (n_samples, n_levels) where n_levels is the number of signature levels
    a: float, parameter for psi function used in the tensor normalisation to get the characteristic signature kernel
    C: float, as above
    '''
    normsq_levels = matrix_diag(gram_matrix).T
    n_samples = normsq_levels.shape[0]
    normsq = np.sum(normsq_levels, axis=1)
    norm_condition = normsq > C
    λ = np.ones(n_samples)
    for i in range(n_samples):
        if norm_condition[i]:
            λ[i] = brentq(norm_func, 0, 1, args=(normsq_levels[i], a, C))
    return λ

def sum_normalise_gram_matrix(K, λ_X, λ_Y, n_levels):
    '''
    # normalise the gram matrix of a signature kernel at each signature level then sum
    '''
    m_λ = (λ_X[:,np.newaxis] @ λ_Y[np.newaxis,:]) ** np.arange(n_levels+1)[:, np.newaxis, np.newaxis]
    K = np.sum(m_λ * K, axis=0)
    return K

def get_gram_matrices(X, Y, kernel, n_levels, a, C):
    K_XX = kernel(X,X)
    λ_X = get_normalisation_constants(K_XX, a, C)
    K_XX = sum_normalise_gram_matrix(K_XX, λ_X, λ_X, n_levels)

    K_YY = kernel(Y,Y)
    λ_Y = get_normalisation_constants(K_YY, a, C)
    K_YY = sum_normalise_gram_matrix(K_YY, λ_Y, λ_Y, n_levels)

    K_XY = kernel(X,Y)
    K_XY = sum_normalise_gram_matrix(K_XY, λ_X, λ_Y, n_levels)

    # zero the diagonals as the sums will include the diagonals and we need kernel(x,x) excluded i.e. kernel of the same sample
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)

    return K_XX, K_YY, K_XY

def get_permutation_indices(n, m):
    '''
    get indices for permutation test
    n: int, number of samples in first sample (X)
    m: int, number of samples in second sample (Y)
    '''
    p = np.random.permutation(n+m)
    sample_1 = p[:n]
    X_inds_sample_1 = sample_1[sample_1 < n]
    Y_inds_sample_1 = sample_1[sample_1 >= n] - n
    sample_2 = p[n:]
    X_inds_sample_2 = sample_2[sample_2 < n]
    Y_inds_sample_2 = sample_2[sample_2 >= n] - n

    return X_inds_sample_1, Y_inds_sample_1, X_inds_sample_2, Y_inds_sample_2

def get_permuted_kernel_sum(K_XX, K_YY, K_XY, inds_X, inds_Y):
    '''
    Calculate the sum of the permuted gram matrix
    gram_X: np.array of shape (n, n)
    gram_Y: np.array of shape (m, m)
    inds_X: 1D np.array up to size n and containing integers in [0,n]
    inds_Y: 1D np.array up to size m and containing integers in [0,m]
    '''
    if len(inds_X) == 0:
        return np.sum(K_YY)
    elif len(inds_Y) == 0:
        return np.sum(K_XX)
    else:
        return np.sum(K_XX[np.ix_(inds_X, inds_X)]) + np.sum(K_YY[np.ix_(inds_Y, inds_Y)]) + 2 * np.sum(K_XY[np.ix_(inds_X, inds_Y)])

def get_permuted_cross_kernel_sum(K_XX, K_YY, K_XY, inds_X_1, inds_Y_1, inds_X_2, inds_Y_2):
    '''
    Calculate the sum of the permuted cross gram matrix
    '''
    if len(inds_X_1) == 0:
        assert len(inds_Y_2) == 0, 'if inds_X_1 is empty then inds_Y_2 must also be empty'
        return np.sum(K_XY)
    elif len(inds_Y_1) == 0:
        assert len(inds_X_2) == 0, 'if inds_Y_1 is empty then inds_X_2 must also be empty'
        return np.sum(K_XY)
    else:
        return (np.sum(K_XX[np.ix_(inds_X_1, inds_X_2)]) +
                np.sum(K_YY[np.ix_(inds_Y_1, inds_Y_2)]) +
                np.sum(K_XY[np.ix_(inds_X_1, inds_Y_2)]) +
                np.sum(K_XY[np.ix_(inds_X_2, inds_Y_1)]))

def sig_kernel_test(X, Y, n_levels, static_kernel, num_permutations=1000, a=1, C=4, stats_plot=True, percentile=0.9, ratio_plot=True, n_steps=10):
    '''
    X: np.array of shape (n_samples, n_features)
    Y: np.array of shape (n_samples, n_features)
    n_levels: int, number of levels of the signature kernel where level 0 is not included e.g. 3 means signature (t0, t1, t2, t3)
    static_kernel: static kernel to be lifted to sequence kernel e.g. LinearKernel or RBFKernel
    percentile: float, percentile to be used for the test
    num_permutations: int, number of permutations to be used to generate the null distribution
    a: float, parameter for psi function used in the tensor normalisation to get the characteristic signature kernel
    C: float, as above
    '''
    assert X.ndim == Y.ndim

    kernel = SignatureKernel(n_levels=n_levels, order=n_levels, normalization=3, static_kernel=static_kernel)

    # calculate Gram matrices with normalisation and diagonal of XX/YY zeroed
    K_XX, K_YY, K_XY = get_gram_matrices(X, Y, kernel, n_levels, a, C)

    # unbiased MMD statistic (could also use biased, doesn't matter if we use permutation tests)
    n = len(K_XX)
    m = len(K_YY)
    mmd = np.sum(K_XX) / (n*(n-1))  + np.sum(K_YY) / (m*(m-1))  - 2*np.sum(K_XY)/(n*m)

    null_mmds = np.zeros(num_permutations)
    for i in range(num_permutations):
        X_inds_sample_1, Y_inds_sample_1, X_inds_sample_2, Y_inds_sample_2 = get_permutation_indices(n, m)
        perm_K_XX_sum = get_permuted_kernel_sum(K_XX, K_YY, K_XY, X_inds_sample_1, Y_inds_sample_1)
        perm_K_YY_sum = get_permuted_kernel_sum(K_XX, K_YY, K_XY, X_inds_sample_2, Y_inds_sample_2)
        perm_K_XY_sum = get_permuted_cross_kernel_sum(K_XX, K_YY, K_XY, X_inds_sample_1, Y_inds_sample_1, X_inds_sample_2, Y_inds_sample_2)
        null_mmds[i] = perm_K_XX_sum / (n*(n-1))  + perm_K_YY_sum / (m*(m-1))  - 2*perm_K_XY_sum/(n*m)

    if stats_plot:
        plot_permutation_samples(null_mmds, statistic=mmd, percentile=percentile)

    if ratio_plot:
        plt.figure()
        mmd_splits = np.empty((2, n_steps+1))
        for i in range(n_steps+1):
            split = i / n_steps
            split_x = int(split * n)
            split_y = int(split * m)
            X_inds_sample_1 = np.arange(split_x)
            Y_inds_sample_1 = np.arange(split_y, m)
            X_inds_sample_2 = np.arange(split_x, n)
            Y_inds_sample_2 = np.arange(split_y)

            perm_K_XX_sum = get_permuted_kernel_sum(K_XX, K_YY, K_XY, X_inds_sample_1, Y_inds_sample_1)
            perm_K_YY_sum = get_permuted_kernel_sum(K_XX, K_YY, K_XY, X_inds_sample_2, Y_inds_sample_2)
            perm_K_XY_sum = get_permuted_cross_kernel_sum(K_XX, K_YY, K_XY, X_inds_sample_1, Y_inds_sample_1, X_inds_sample_2, Y_inds_sample_2)

            mmd_splits[0, i] = split
            mmd_splits[1, i] = perm_K_XX_sum / (n*(n-1))  + perm_K_YY_sum / (m*(m-1))  - 2*perm_K_XY_sum/(n*m)

        plt.plot(mmd_splits[0], mmd_splits[1])
        plt.axhline(y=mmd, c='r')
        legend = ['MMD at different split ratios', 'Actual test statistic']
        plt.legend(legend)

    return mmd, null_mmds

def mmd_permutation_ratio_plot(X, Y, n_levels, static_kernel, n_steps=10, a=1, C=4):
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
    K_XX, K_YY, K_XY = get_gram_matrices(X, Y, kernel, n_levels, a, C)

    # unbiased MMD statistic (could also use biased, doesn't matter if we use permutation tests)
    n = len(K_XX)
    m = len(K_YY)
    mmd = np.sum(K_XX) / (n*(n-1))  + np.sum(K_YY) / (m*(m-1))  - 2*np.sum(K_XY)/(n*m)

    mmd_splits = np.empty((2, n_steps+1))
    for i in range(n_steps+1):
        split = i / n_steps
        split_x = int(split * n)
        split_y = int(split * m)
        X_inds_sample_1 = np.arange(split_x)
        Y_inds_sample_1 = np.arange(split_y, m)
        X_inds_sample_2 = np.arange(split_x, n)
        Y_inds_sample_2 = np.arange(split_y)

        perm_K_XX_sum = get_permuted_kernel_sum(K_XX, K_YY, K_XY, X_inds_sample_1, Y_inds_sample_1)
        perm_K_YY_sum = get_permuted_kernel_sum(K_XX, K_YY, K_XY, X_inds_sample_2, Y_inds_sample_2)
        perm_K_XY_sum = get_permuted_cross_kernel_sum(K_XX, K_YY, K_XY, X_inds_sample_1, Y_inds_sample_1, X_inds_sample_2, Y_inds_sample_2)

        mmd_splits[0, i] = split
        mmd_splits[1, i] = perm_K_XX_sum / (n*(n-1))  + perm_K_YY_sum / (m*(m-1))  - 2*perm_K_XY_sum/(n*m)

    plt.plot(mmd_splits[0], mmd_splits[1])
    plt.axhline(y=mmd, c='r')
    legend = ['MMD at different split ratios', 'Actual test statistic']
    plt.legend(legend)

def plot_permutation_samples(null_samples, statistic=None, percentile=0.9, two_tailed=False):
    plt.hist(null_samples, bins=100)

    if two_tailed:
        plt.axvline(x=np.percentile(null_samples, 50 * (1 + percentile)), c='b')
    else:
        plt.axvline(x=np.percentile(null_samples, 100*percentile), c='b')
    legend = [f'{int(100*percentile)} percentile']

    if statistic is not None:
        percentile = (null_samples < statistic).sum() / len(null_samples)
        plt.axvline(x=statistic, c='r')
        legend += [f'Test statistic at {int(percentile*100)} percentile']

    plt.legend(legend)
    plt.xlabel('Test statistic value')
    plt.ylabel('Counts')

# simulate geometric Brownian motion paths
def gen_GBM_path(mu, sigma, dt, n_paths, seq_len, seed=None):
    rng = np.random.default_rng(seed)
    n_steps = seq_len - 1
    path = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal((n_paths, n_steps)))
    path = np.cumprod(path, axis=1)
    path = np.concatenate([np.ones((n_paths, 1)), path], axis=1)
    path = path[..., np.newaxis]
    return path # shape (n_paths, seq_len, 1)

def gen_quantlib_paths(process, dt, n_paths, seq_len, seed, return_all_paths):

    times = ql.TimeGrid((seq_len-1)*dt, seq_len-1) # creates list of times starting from 0 to (seq_len-1)*dt with step size dt
    dimension = process.factors() # 2 factors for Heston model i.e. spot and vol

    randomGenerator = ql.UniformRandomGenerator() if seed is None else ql.UniformRandomGenerator(seed=seed) # seed of 0 seems to not set a seed
    rng = ql.UniformRandomSequenceGenerator(dimension * (seq_len-1), randomGenerator)
    sequenceGenerator = ql.GaussianRandomSequenceGenerator(rng)
    pathGenerator = ql.GaussianMultiPathGenerator(process, list(times), sequenceGenerator, False)

    paths = [[] for i in range(dimension)]
    for _ in range(n_paths):
        samplePath = pathGenerator.next()
        values = samplePath.value()

        for j in range(dimension):
            paths[j].append([x for x in values[j]])

    if return_all_paths:
        return np.array(paths).transpose(1,0,2)
    else:
        return np.array(paths[0])[..., np.newaxis]

# simulate Heston paths
def gen_Heston_path(mu, v0, kappa, theta, rho, sigma, dt, n_paths, seq_len, seed=None, return_vols=False):
    today = ql.Date().todaysDate()
    riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today, mu, ql.Actual365Fixed()))
    dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.00, ql.Actual365Fixed()))
    initialValue = ql.QuoteHandle(ql.SimpleQuote(1.0))

    hestonProcess = ql.HestonProcess(riskFreeTS, dividendTS, initialValue, v0, kappa, theta, sigma, rho)

    return gen_quantlib_paths(hestonProcess, dt, n_paths, seq_len, seed=seed, return_all_paths=return_vols)

def gen_OU_path(kappa, theta, sigma, dt, n_paths, seq_len, seed=None):
    process = ql.ExtendedOrnsteinUhlenbeckProcess(kappa, sigma, 1.0, lambda x: theta)
    return gen_quantlib_paths(process, dt, n_paths, seq_len, seed=seed, return_all_paths=False)
