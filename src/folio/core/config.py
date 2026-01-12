"""Configuration dataclasses for projects."""

from dataclasses import dataclass, field
from typing import Any, Literal

from folio.exceptions import InvalidSchemaError


@dataclass
class TargetConfig:
    """Configuration for a single optimization target.

    Defines how to extract an optimization target from observations. Each
    TargetConfig represents one objective in the optimization problem.

    For direct targets, use `objective` to specify the output name.
    For derived targets (ratio, difference, slope), provide the required
    parameters for that target type.

    Parameters
    ----------
    name : str | None, optional
        Optional label or description for this target configuration.
        Not used for computation, only for display/documentation purposes.
    objective : str | None, optional
        The output name to optimize (for "direct" targets) or a descriptive
        name for derived targets. Required for direct targets.
    objective_mode : str, optional
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
        If validation fails: invalid objective_mode or missing required fields.

    Examples
    --------
    Direct target (maximize yield):

    >>> direct = TargetConfig(objective="yield", objective_mode="maximize")

    Ratio target (selectivity):

    >>> ratio = TargetConfig(
    ...     objective="selectivity",
    ...     target_type="ratio",
    ...     numerator="product_a",
    ...     denominator="product_b",
    ... )

    Minimize a target:

    >>> minimize = TargetConfig(objective="cost", objective_mode="minimize")
    """

    name: str | None = None
    objective: str | None = None
    objective_mode: str = "maximize"

    target_type: Literal["direct", "ratio", "difference", "slope"] = "direct"

    numerator: str | None = None
    denominator: str | None = None

    first: str | None = None
    second: str | None = None

    slope_outputs: list[str] | None = None
    slope_x: list[float] | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate target configuration.

        Raises
        ------
        InvalidSchemaError
            If objective is not set or objective_mode is invalid.
        """
        if self.target_type == "direct" and self.objective is None:
            raise InvalidSchemaError("Direct target requires 'objective' to be set.")
        if self.objective_mode not in ("maximize", "minimize"):
            raise InvalidSchemaError(
                f"objective_mode must be 'maximize' or 'minimize', "
                f"got '{self.objective_mode}'"
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
        Surrogate model for Bayesian optimization: "gp" (SingleTaskGPSurrogate)
        or "multitask_gp" (MultiTaskGPSurrogate). Defaults to "gp".
    acquisition : str, optional
        Acquisition function for single-objective Bayesian optimization:
        "ei" (Expected Improvement) or "ucb" (Upper Confidence Bound).
        Defaults to "ei". Ignored for non-Bayesian recommenders.
    mo_acquisition : str, optional
        Acquisition function for multi-objective Bayesian optimization:
        "nehvi" (Noisy Expected Hypervolume Improvement).
        Defaults to "nehvi". Ignored for single-objective or non-Bayesian.
    n_initial : int, optional
        Number of initial random samples before using the surrogate model.
        Defaults to 5. The surrogate needs sufficient data to make useful predictions.
    surrogate_kwargs : dict[str, Any], optional
        Keyword arguments passed to the surrogate model constructor.
        For SingleTaskGPSurrogate: {"kernel": "matern", "nu": 2.5, "ard": True,
        "normalize_inputs": True, "normalize_outputs": True}.
    acquisition_kwargs : dict[str, Any], optional
        Keyword arguments passed to the acquisition function constructor.
        For EI: {"xi": 0.01}. For UCB: {"beta": 2.0}. For NEHVI: {"alpha": 0.0}.

    Examples
    --------
    >>> bo_config = RecommenderConfig(type="bayesian", surrogate="gp", acquisition="ei")
    >>> random_config = RecommenderConfig(type="random", n_initial=10)
    >>> mo_config = RecommenderConfig(
    ...     surrogate="multitask_gp",
    ...     mo_acquisition="nehvi",
    ... )
    """

    type: str = "bayesian"
    surrogate: str = "gp"
    acquisition: str = "ei"
    mo_acquisition: str = "nehvi"
    n_initial: int = 5
    surrogate_kwargs: dict[str, Any] = field(default_factory=dict)
    acquisition_kwargs: dict[str, Any] = field(default_factory=dict)
