"""Abstract base class for surrogate models."""

from abc import ABC, abstractmethod

import numpy as np


class Surrogate(ABC):
    """Abstract base class for surrogate models used in Bayesian optimization.

    Surrogates are probabilistic models that approximate the objective function.
    They provide both point predictions (mean) and uncertainty estimates (std)
    for candidate points, enabling acquisition functions to balance exploration
    and exploitation.

    The interface follows sklearn conventions: `fit` returns `self` to enable
    method chaining, and `predict` returns a tuple of (mean, std) arrays.

    Attributes
    ----------
    _is_fitted : bool
        Internal flag tracking whether fit() has been called. Subclasses should
        set this to True at the end of their fit() implementation.

    Notes
    -----
    Subclasses must:

    - Call `super().__init__()` in their `__init__` method
    - Implement `fit` to train on observations and return `self`
    - Implement `predict` to return (mean, std) arrays
    - Raise `NotFittedError` if `predict` is called before `fit`
    - Set `self._is_fitted = True` after successful fitting

    Examples
    --------
    Typical usage pattern:

    >>> surrogate = GPSurrogate(kernel="rbf")
    >>> surrogate.fit(X_train, y_train).predict(X_test)
    (array([0.5, 0.8]), array([0.1, 0.2]))

    Method chaining:

    >>> mean, std = GPSurrogate().fit(X_train, y_train).predict(X_test)

    Subclass implementation pattern:

    >>> class MySurrogate(Surrogate):
    ...     def __init__(self):
    ...         super().__init__()
    ...
    ...     def fit(self, X: np.ndarray, y: np.ndarray) -> "Surrogate":
    ...         # Train model on X, y
    ...         self._is_fitted = True
    ...         return self
    ...
    ...     def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...         if not self._is_fitted:
    ...             raise NotFittedError("Call fit() before predict()")
    ...         # Return predictions
    ...         return mean, std
    """

    def __init__(self):
        """Initialize the surrogate with unfitted state."""
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Surrogate":
        """Fit the surrogate model to training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training input features. Each row is an observation, each column
            is a feature (input dimension).
        y : np.ndarray, shape (n_samples,)
            Training target values. The scalar objective value for each
            observation.

        Returns
        -------
        Surrogate
            Returns self for method chaining (sklearn convention).

        Raises
        ------
        ValueError
            If X and y have incompatible shapes (different number of samples).

        Notes
        -----
        Implementations should set `self._is_fitted = True` after successful
        fitting to enable the fitted state check in `predict`.

        Examples
        --------
        >>> surrogate = GPSurrogate()
        >>> surrogate.fit(X_train, y_train)
        <GPSurrogate object>

        Method chaining:

        >>> mean, std = surrogate.fit(X_train, y_train).predict(X_test)
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and standard deviation at candidate points.

        Parameters
        ----------
        X : np.ndarray, shape (n_candidates, n_features)
            Candidate points to evaluate. Each row is a candidate, columns
            must match the features used in `fit`.

        Returns
        -------
        mean : np.ndarray, shape (n_candidates,)
            Predicted mean value at each candidate point.
        std : np.ndarray, shape (n_candidates,)
            Predicted standard deviation (uncertainty) at each candidate point.
            Values should be non-negative.

        Raises
        ------
        NotFittedError
            If called before `fit`. Implementations must check the fitted state
            and raise this error with a helpful message.
        ValueError
            If X has wrong number of features (doesn't match training data).

        Notes
        -----
        The uncertainty estimate (std) is critical for Bayesian optimization.
        Points with high uncertainty are candidates for exploration, while
        points with high predicted mean are candidates for exploitation.

        Examples
        --------
        >>> surrogate.fit(X_train, y_train)
        >>> mean, std = surrogate.predict(X_test)
        >>> mean.shape
        (100,)
        >>> std.shape
        (100,)
        >>> np.all(std >= 0)
        True
        """
        ...
