from __future__ import annotations

from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn.base import BaseEstimator

from .. import utils
from ..utils import ArrayOnCPU, RandomStateOrSeed

from typing import Optional


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class Kernel(BaseEstimator, metaclass=ABCMeta):
    """Base class for Kernels.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def fit(self, X : ArrayOnCPU, y : Optional[ArrayOnCPU] = None) -> Kernel:
        raise NotImplementedError

    @abstractmethod
    def _K(self, X : ArrayOnCPU, Y : Optional[ArrayOnCPU] = None) -> ArrayOnCPU:
        pass

    @abstractmethod
    def _Kdiag(self, X : ArrayOnCPU) -> ArrayOnCPU:
        pass

    def __call__(self, X : ArrayOnCPU, Y : Optional[ArrayOnCPU] = None, diag : bool = False, return_on_gpu : bool = False) -> ArrayOnCPU:
        X = np.asarray(X)
        Y = np.asarray(Y) if Y is not None else None
        if diag:
            K = self._Kdiag(X)
        else:
            K =  self._K(X, Y)
        return K

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class LinearKernel(Kernel):
    """Class for linear (static) kernel."""

    def __init__(self, sigma : float = 1.0) -> None:
        self.sigma = utils.check_positive_value(sigma, 'sigma')

    def _K(self, X : ArrayOnCPU, Y : Optional[ArrayOnCPU] = None) -> ArrayOnCPU:
        return self.sigma**2 * utils.matrix_mult(X, Y, transpose_Y=True)

    def _Kdiag(self, X : ArrayOnCPU) -> ArrayOnCPU:
        return self.sigma**2 * utils.squared_norm(X, axis=-1)

class PolynomialKernel(Kernel):
    """Class for polynomial (static) kernel."""

    def __init__(self, sigma : float = 1.0, degree : float = 3.0, gamma : float = 1.0) -> None:
        self.sigma = utils.check_positive_value(sigma, 'sigma')
        self.degree = utils.check_positive_value(degree, 'degree')
        self.gamma = gamma

    def _K(self, X : ArrayOnCPU, Y : Optional[ArrayOnCPU] = None) -> ArrayOnCPU:
        return self.sigma**2 * np.power(utils.matrix_mult(X, Y, transpose_Y=True) + self.gamma, self.degree)

    def _Kdiag(self, X : ArrayOnCPU) -> ArrayOnCPU:
        return self.sigma**2 * np.power(utils.squared_norm(X, axis=-1) + self.gamma, self.degree)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class RBFKernelMix(Kernel):
    """Mixture of RBF kernels"""

    def __init__(self, sigma : float = 1.0, lengthscale : list = [0.1, 0.5, 1.0]) -> None:
        self.sigma = utils.check_positive_value(sigma, 'sigma')
        self.lengthscale = np.asarray(lengthscale)[np.newaxis, np.newaxis, :]

    def _K(self, X : ArrayOnCPU, Y : Optional[ArrayOnCPU] = None) -> ArrayOnCPU:
        D2_scaled = utils.squared_euclid_dist(X, Y)[..., np.newaxis] / self.lengthscale**2
        # print(self.lengthscale.shape, utils.squared_euclid_dist(X, Y)[..., np.newaxis].shape, D2_scaled.shape)
        return self.sigma**2 * np.sum(np.exp(-D2_scaled), axis=-1)

    def _Kdiag(self, X : ArrayOnCPU) -> ArrayOnCPU:
        return np.full((X.shape[0],), self.sigma**2)

class StationaryKernel(Kernel):
    """Base class for stationary (static) kernels.

    Warning: This class should not be used directly.
    Use derived classes instead."""

    def __init__(self, sigma : float = 1.0, lengthscale : float = 1.0) -> None:
        self.sigma = utils.check_positive_value(sigma, 'sigma')
        self.lengthscale = utils.check_positive_value(lengthscale, 'lengthscale')

    def _Kdiag(self, X : ArrayOnCPU) -> ArrayOnCPU:
        return np.full((X.shape[0],), self.sigma**2)

class RBFKernel(StationaryKernel):
    """Radial Basis Function aka Causs (static) kernel ."""

    def _K(self, X : ArrayOnCPU, Y : Optional[ArrayOnCPU] = None) -> ArrayOnCPU:
        D2_scaled = utils.squared_euclid_dist(X, Y) / self.lengthscale**2
        return self.sigma**2 * np.exp(-D2_scaled)

class Matern12Kernel(StationaryKernel):
    """Matern12 (static) kernel ."""

    def _K(self, X : ArrayOnCPU, Y : Optional[ArrayOnCPU] = None) -> ArrayOnCPU:
        D_scaled = utils.euclid_dist(X, Y) / self.lengthscale
        return self.sigma**2 * np.exp(-D_scaled)

class Matern32Kernel(StationaryKernel):
    """Matern32 (static) kernel ."""

    def _K(self, X : ArrayOnCPU, Y : Optional[ArrayOnCPU] = None) -> ArrayOnCPU:
        sqrt3 = np.sqrt(3.)
        D_scaled = sqrt3 * utils.euclid_dist(X, Y) / self.lengthscale
        return self.sigma**2 * (1. + D_scaled) * np.exp(-D_scaled)

class Matern52Kernel(StationaryKernel):
    """Matern52 (static) kernel ."""

    def _K(self, X : ArrayOnCPU, Y : Optional[ArrayOnCPU] = None) -> ArrayOnCPU:
        D2_scaled = 5 * utils.squared_euclid_dist(X, Y) / self.lengthscale**2
        D_scaled = utils.robust_sqrt(D2_scaled)
        return self.sigma**2 * (1. + D_scaled + D2_scaled / 3.) * np.exp(-D_scaled)

class RationalQuadraticKernel(StationaryKernel):
    """Rational Quadratic (static) kernel ."""

    def __init__(self, sigma : float = 1.0, lengthscale : float = 1.0, alpha : float = 1.0) -> None:
        super().__init__(sigma=sigma, lengthscale=lengthscale)
        self.alpha = utils.check_positive_value(alpha, 'alpha')

    def _K(self, X : ArrayOnCPU, Y : Optional[ArrayOnCPU] = None) -> ArrayOnCPU:
        D2_scaled = utils.squared_euclid_dist(X, Y) / (2 * self.alpha * self.lengthscale**2)
        return self.sigma**2 * np.power((1 + D2_scaled), -self.alpha)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------