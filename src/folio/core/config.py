"""Configuration dataclasses for projects."""

from dataclasses import dataclass, field
from typing import Any, Literal

from folio.exceptions import InvalidSchemaError


@dataclass
class TargetConfig:
    """Configuration for the optimization target.

    Defines how to extract optimization targets from observations. Supports both
    single-objective (scalar target) and multi-objective optimization.

    For single-objective: use `objective` to specify the output name (for direct
    targets) or provide derived target parameters (numerator/denominator, etc.).

    For multi-objective: use `objectives` to specify multiple output names and
    `reference_point` for hypervolume calculation.

    Parameters
    ----------
    name : str | None, optional
        Optional label or description for this target configuration.
        Not used for computation, only for display/documentation purposes.
    objective : str | None, optional
        For single-objective optimization: the output name to optimize (for "direct"
        targets) or a descriptive name for derived targets. Required when
        `objectives` is None.
    objective_mode : str, optional
        Optimization direction for single-objective: "maximize" or "minimize".
        Defaults to "maximize".
    objectives : list[str] | None, optional
        For multi-objective optimization: list of output names to optimize.
        When set, enables multi-objective mode.
    objective_modes : list[str] | None, optional
        Optimization directions for each objective in multi-objective mode.
        Must have same length as `objectives`. Defaults to all "maximize" if None.
        Each element must be "maximize" or "minimize".
    reference_point : list[float] | None, optional
        Reference point for hypervolume calculation in multi-objective optimization.
        Required when `objectives` is set. Should be dominated by all Pareto-optimal
        points (i.e., worse than the worst expected objective values).
    target_type : {"direct", "ratio", "difference", "slope"}, optional
        How to compute the target value for single-objective optimization.
        Defaults to "direct".
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
        If validation fails: invalid objective_mode, missing required fields,
        or mismatched list lengths.

    Examples
    --------
    Single-objective direct target:

    >>> direct = TargetConfig(objective="yield", objective_mode="maximize")

    Single-objective ratio target:

    >>> ratio = TargetConfig(
    ...     objective="selectivity",
    ...     target_type="ratio",
    ...     numerator="product_a",
    ...     denominator="product_b",
    ... )

    Multi-objective optimization:

    >>> multi = TargetConfig(
    ...     objectives=["yield", "purity"],
    ...     objective_modes=["maximize", "maximize"],
    ...     reference_point=[0.0, 0.0],
    ... )
    """

    name: str | None = None
    objective: str | None = None
    objective_mode: str = "maximize"

    objectives: list[str] | None = None
    objective_modes: list[str] | None = None
    reference_point: list[float] | None = None

    target_type: Literal["direct", "ratio", "difference", "slope"] = "direct"

    numerator: str | None = None
    denominator: str | None = None

    first: str | None = None
    second: str | None = None

    slope_outputs: list[str] | None = None
    slope_x: list[float] | None = None

    def __post_init__(self) -> None:
        if self.is_multiobjective():
            self._validate_multiobjective()
        else:
            self._validate_single_objective()

    def _validate_single_objective(self) -> None:
        """Validate single-objective configuration.

        Raises
        ------
        InvalidSchemaError
            If objective is not set or objective_mode is invalid.
        """
        if self.objective is None:
            raise InvalidSchemaError(
                "Single-objective optimization requires 'objective' to be set. "
                "For multi-objective, set 'objectives' instead."
            )
        if self.objective_mode not in ("maximize", "minimize"):
            raise InvalidSchemaError(
                f"objective_mode must be 'maximize' or 'minimize', "
                f"got '{self.objective_mode}'"
            )

    def _validate_multiobjective(self) -> None:
        """Validate multi-objective configuration.

        Raises
        ------
        InvalidSchemaError
            If reference_point is missing or objective_modes length doesn't match.
        """
        if self.reference_point is None:
            raise InvalidSchemaError(
                "Multi-objective optimization requires 'reference_point' to be set "
                "for hypervolume calculation."
            )

        if self.objective_modes is None:
            object.__setattr__(
                self, "objective_modes", ["maximize"] * len(self.objectives)
            )
        elif len(self.objective_modes) != len(self.objectives):
            raise InvalidSchemaError(
                f"Length of 'objective_modes' ({len(self.objective_modes)}) must match "
                f"length of 'objectives' ({len(self.objectives)})."
            )

        for mode in self.objective_modes:
            if mode not in ("maximize", "minimize"):
                raise InvalidSchemaError(
                    f"Each objective_mode must be 'maximize' or 'minimize', "
                    f"got '{mode}'"
                )

    def is_multiobjective(self) -> bool:
        """Check if this configuration is for multi-objective optimization.

        Returns
        -------
        bool
            True if `objectives` is set (multi-objective mode),
            False otherwise (single-objective mode).
        """
        return self.objectives is not None


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
        Ignored for non-Bayesian recommenders.
    acquisition : str, optional
        Acquisition function for Bayesian optimization: "ei" (Expected Improvement)
        or "ucb" (Upper Confidence Bound). Defaults to "ei".
        Ignored for non-Bayesian recommenders.
    n_initial : int, optional
        Number of initial random samples before using the surrogate model.
        Defaults to 5. The surrogate needs sufficient data to make useful predictions.
    surrogate_kwargs : dict[str, Any], optional
        Keyword arguments passed to the surrogate model constructor.
        For SingleTaskGPSurrogate: {"kernel": "matern", "nu": 2.5, "ard": True,
        "normalize_inputs": True, "normalize_outputs": True}.
    acquisition_kwargs : dict[str, Any], optional
        Keyword arguments passed to the acquisition function constructor.
        For EI: {"xi": 0.01}. For UCB: {"beta": 2.0}.

    Examples
    --------
    >>> bo_config = RecommenderConfig(type="bayesian", surrogate="gp", acquisition="ei")
    >>> random_config = RecommenderConfig(type="random", n_initial=10)
    >>> custom_gp = RecommenderConfig(
    ...     surrogate="gp",
    ...     surrogate_kwargs={"kernel": "rbf", "ard": False},
    ...     acquisition_kwargs={"xi": 0.1},
    ... )
    """

    type: str = "bayesian"
    surrogate: str = "gp"
    acquisition: str = "ei"
    n_initial: int = 5
    surrogate_kwargs: dict[str, Any] = field(default_factory=dict)
    acquisition_kwargs: dict[str, Any] = field(default_factory=dict)
