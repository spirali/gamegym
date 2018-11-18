from ..game import Game, GameState
from ..utils import get_rng
from ..distribution import Explicit
import numpy as np


class LinearValueStore:
    """
    A store for an `np.array` of parameters.
    
    A value is a linear combination of features and parameters.
    Update is performed along a (possibly sparse) gradient.
    If `fixed_mean` is set, it is enforced on every update (which may be slow for large shapes).
    Optional on-demand mean, L1 and L2 normalizations.
    """
    def __init__(self, initializer=None, *, shape=None, dtype='f',
                 fix_mean=None, normalize_mean=None,
                 normalize_l1=None, normalize_l2=None):
        """
        The initial values are either taken from `initializer` (incl. dtype)
        or set to normalize_mean, or force_mean, or 0.0.
        """
        if initializer is not None:
            assert isinstance(initializer, np.ndarray)
            assert shape is None or tuple(shape) == initializer.shape
            self.values = initializer.copy()
        else:
            self.values = np.full(shape, fix_mean or normalize_mean or 0.0, dtype=dtype)
        self.normalize_mean = normalize_mean
        self.normalize_l1 = normalize_l1
        self.normalize_l2 = normalize_l2
        self.fix_mean = fix_mean
        if self.fix_mean is not None:
            self.values += (self.fix_mean - np.mean(self.values))

    def get(self, features):
        """
        Return the value corresponding to features (which may be a sparse array).
        """
        return np.tensordot(features, self.values, axes=len(self.values.shape))[()]

    def update(self, features, gradient):
        """
        Update the values corresponding to features by the gradien (which should be scalar).

        Always normalizes parameter mean if `fix_mean` is set.
        """
        self.values += np.multiply(features, gradient)
        if self.fix_mean is not None:
            self.values += (self.fix_mean - np.mean(self.values))

    def regularize(self, step=1e-3):
        """
        Applies gradual normalization of the mean, L1 and L2 norms (if any are set)
        with the given step size.
        """
        # normalize the mean - by step size
        if self.normalize_mean is not None:
            self.values += (self.normalize_mean - np.mean(self.values)) * step

        # TODO(gavento): additive vs multiplicative norm updates

        # normalize the L1 norm - by step size
        if self.normalize_l1 is not None:
            l1 = np.sum(np.abs(self.values))
            self.values *= 1.0 + (self.normalize_l1 - l1) * step

        # normalize the L2 norm - by step size
        if self.normalize_l2 is not None:
            l2 = np.linalg.norm(self.values)
            self.values *= 1.0 + (self.normalize_l2 - l2) * step

