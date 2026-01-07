"""Configuration dataclasses for projects."""

from dataclasses import dataclass, field
from typing import Any, Literal

from folio.exceptions import InvalidSchemaError


@dataclass
class TargetConfig:
    """Configuration for the optimization target.

    Defines how to extract a scalar optimization target from observations.
    Supports direct output values, ratios, differences, and slope calculations.

    Parameters
    ----------
    name : str
        For "direct" targets, the output name to optimize. For other target types,
        a descriptive name for the derived metric.
    mode : str, optional
        Optimization direction: "maximize" or "minimize". Defaults to "maximize".
    target_type : {"direct", "ratio", "difference", "slope"}, optional
        How to compute the target value. Defaults to "direct".
    numerator : str | None, optional
        Output name for ratio numerator. Required when target_type is "ratio".
    denominator : str | None, optional
        Output name for ratio denominator. Required when target_type is "ratio".
    first : str | None, optional
        Output name for first term in difference. Required when
        target_type is "difference".
    second : str | None, optional
        Output name for second term in difference. Required when
        target_type is "difference".
    slope_outputs : list[str] | None, optional
        Output names for slope calculation (y-values). Required when
        target_type is "slope". Must have at least 3 outputs.
    slope_x : list[float] | None, optional
        X-values for slope calculation. Required when target_type is "slope".
        Must have same length as slope_outputs.

    Raises
    ------
    InvalidSchemaError
        If mode is not "maximize" or "minimize".

    Examples
    --------
    >>> direct = TargetConfig("yield", mode="maximize")
    >>> ratio = TargetConfig("selectivity", target_type="ratio",
    ...                      numerator="product_a", denominator="product_b")
    """

    name: str
    mode: str = "maximize"
    target_type: Literal["direct", "ratio", "difference", "slope"] = "direct"

    numerator: str | None = None
    denominator: str | None = None

    first: str | None = None
    second: str | None = None

    slope_outputs: list[str] | None = None
    slope_x: list[float] | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("maximize", "minimize"):
            raise InvalidSchemaError(
                f"Target mode must be 'maximize' or 'minimize', got '{self.mode}'"
            )


@dataclass
class RecommenderConfig:
    """Configuration for the experiment recommender.

    Defines how the recommender suggests next experiments. Supports Bayesian
    optimization with configurable surrogate models and acquisition functions,
    as well as simpler strategies like random or grid search.

    Parameters
    ----------
    type : str, optional
        Recommender strategy: "bayesian", "random", or "grid". Defaults to "bayesian".
    surrogate : str, optional
        Surrogate model for Bayesian optimization: "gp" (Gaussian Process).
        Defaults to "gp". Ignored for non-Bayesian recommenders.
    acquisition : str, optional
        Acquisition function for Bayesian optimization: "ei" (Expected Improvement)
        or "ucb" (Upper Confidence Bound). Defaults to "ei".
        Ignored for non-Bayesian recommenders.
    n_initial : int, optional
        Number of initial random samples before using the surrogate model.
        Defaults to 5. The surrogate needs sufficient data to make useful predictions.
    kwargs : dict[str, Any], optional
        Additional keyword arguments passed to the recommender implementation.
        For example, {"beta": 2.0} for UCB acquisition function.

    Examples
    --------
    >>> bo_config = RecommenderConfig(type="bayesian", surrogate="gp", acquisition="ei")
    >>> random_config = RecommenderConfig(type="random", n_initial=10)
    """

    type: str = "bayesian"
    surrogate: str = "gp"
    acquisition: str = "ei"
    n_initial: int = 5
    kwargs: dict[str, Any] = field(default_factory=dict)
