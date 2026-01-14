# CLAUDE.md

## Project Overview

Folio is an open-source electronic lab notebook with intelligent experiment suggestions for chemistry labs. Human-in-the-loop MVP, with architecture enabling closed-loop automation.

Target users: lab scientists. No coding required (GUI), but API available for those who want it.

## High-Level Architecture: Three Bishops

Folio uses a "King and Three Bishops" architecture for clear separation of concerns:
```
┌─────────────────────────────────────────────────────────────┐
│                        Folio (API)                          │
│                          King                               │
│         Top-level orchestrator, user's entry point          │
└─────────────────────────────┬───────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   Project     │     │  Recommender  │     │   Executor    │
│    (ELN)      │     │     (BO)      │     │  (Automation) │
│   Bishop 1    │     │   Bishop 2    │     │   Bishop 3    │
├───────────────┤     ├───────────────┤     ├───────────────┤
│ Schema        │     │ Surrogate     │     │ HumanExecutor │
│ Observations  │     │ Acquisition   │     │ ClaudeLight   │
│ Validation    │     │ KABO          │     │ Opentrons     │
│ Export        │     │ Batch BO      │     │ Custom APIs   │
│ LLM search    │     │ Active learn  │     │               │
│ Cloud sync    │     │               │     │               │
│ Dashboards    │     │               │     │               │
│ Safety checks │     │               │     │               │
└───────────────┘     └───────────────┘     └───────────────┘
      Data               Intelligence           Action
  "what happened"       "what to try"         "run it"
```

**Shared utilities:** `targets/` and `surrogates/` are top-level modules used by multiple bishops (Project uses targets for training data extraction; Recommender uses surrogates for modeling; both can be used standalone for visualization/analysis).

## MVP Scope

Three pillars, all required:
1. **ELN**: Create projects, record observations (CRUD), attach images, document procedures/hazards
2. **PDF Export**: Quarto-based, professional lab notebook output
3. **BO Suggestions**: Smart next-experiment recommendations

Plus:
- **libSQL Cloud Sync**: Distributed data sharing for collaboration (Claude-Light integration)
- **ClaudeLightExecutor**: Fully autonomous closed-loop demo
- **Documentation**: Proper docstrings (NumPy-style) on all functions

Two interfaces:
- **Streamlit GUI**: Full workflow for non-coders
- **Python API**: Same functionality for Jupyter/Quarto power users

## v1.1 Scope (Post-MVP)

- **KABO Embeddings**: Categorical inputs → LLM description → embedding → PCA → latent GP
- **LLM-informed initialization**: Query LLM for literature-based starting conditions before BO

## Development Workflow

**Test-first approach:**
1. Claude Code writes: test cases + function signatures with docstrings
2. Developer writes: implementation
3. Run pytest → iterate until green

### Implementation Boundaries

By default, Claude Code builds scaffolding but leaves core logic for the developer.

**Claude Code writes:**
- Function/method signatures with type hints
- Docstrings (NumPy-style)
- Import statements
- Test cases
- Class structure and inheritance
- `__init__` methods (storing parameters, basic validation)
- Simple boilerplate (dataclasses, property accessors, `__repr__`, etc.)
- Straightforward glue code

**Claude Code leaves as `...` or `raise NotImplementedError()`:**
- Core algorithm implementations
- Complex business logic
- Non-trivial data transformations

**Exception:** If the user says "IMPLEMENT THIS DIRECTLY" (all caps), write the full implementation.

Example:
```python
# Claude Code writes this
def encode_categorical(value: str, levels: list[str]) -> int:
    """Convert categorical value to integer index.

    Parameters
    ----------
    value : str
        The categorical value to encode.
    levels : list[str]
        Valid levels for this categorical.

    Returns
    -------
    int
        Index of value in levels.

    Raises
    ------
    InvalidInputError
        If value not in levels.
    """
    raise NotImplementedError

# Test
def test_encode_categorical_valid():
    assert encode_categorical("ethanol", ["water", "ethanol", "dmso"]) == 1

def test_encode_categorical_invalid():
    with pytest.raises(InvalidInputError):
        encode_categorical("acetone", ["water", "ethanol", "dmso"])
```

Developer fills in implementation. Tests tell you when you're done.

## Architecture

```
src/folio/
├── core/
│   ├── config.py       # TargetConfig, RecommenderConfig (no circular deps)
│   ├── schema.py       # InputSpec, OutputSpec
│   ├── observation.py  # Observation dataclass
│   ├── project.py      # Project dataclass
│   └── database.py     # SQLite CRUD operations
├── recommenders/       # Recommender interface + implementations
│   ├── base.py         # Recommender ABC with recommend() and recommend_from_data()
│   ├── bayesian.py     # BayesianRecommender (GP + acquisition optimization)
│   ├── random.py       # RandomRecommender (uniform sampling)
│   └── acquisitions/   # Acquisition functions (EI, UCB) - internal to recommenders
├── surrogates/
│   ├── base.py         # Surrogate ABC
│   ├── gp.py           # SingleTaskGPSurrogate (BoTorch, single-output GP)
│   ├── multitask_gp.py # MultiTaskGPSurrogate (BoTorch, multi-output GP with ICM)
│   └── transforms.py   # TaskStandardize (per-task outcome normalization)
├── targets/
│   ├── base.py         # ScalarTarget ABC
│   └── builtin.py      # DirectTarget, RatioTarget, etc.
├── executors/          # Executor interface (HumanExecutor, ClaudeLightExecutor)
├── ui/                 # Streamlit app
├── api.py              # Folio class: main entry point for users
└── exceptions.py       # Custom exceptions
```

Note: `targets/` uses `TYPE_CHECKING` for `Observation` imports to avoid circular dependencies with `core/`.

## Key Abstractions

- **Folio**: Main API class. Orchestrates projects, observations, and recommendations. Caches recommenders per project. Entry point: `Folio(db_path)`.
- **Project**: Experiment schema (inputs, outputs, target_configs list, reference_point, recommender_config). Supports single and multi-objective via `is_multi_objective()`.
- **Observation**: Single data point (inputs dict, outputs dict, timestamp, notes, tag, images, failed)
- **Target**: Extracts scalar optimization target from Observation (direct or derived from outputs)
- **Recommender**: Suggests next experiments. Interface: `recommend(observations) → dict`, `recommend_from_data(X, y, bounds, maximize) → np.ndarray` where `maximize: list[bool]`. Implementations: BayesianRecommender, RandomRecommender
- **Surrogate**: Model that fits observations. Interface: `fit(X, y)`, `predict(X) → (mean, std)`. SingleTaskGPSurrogate for single-output (y shape (n, 1)), MultiTaskGPSurrogate for correlated multi-output.
- **TaskStandardize**: BoTorch OutcomeTransform for per-task standardization in MultiTaskGP. Solves scale imbalance when objectives have different magnitudes (e.g., MW ~10^5 vs PDI ~1-3).
- **Acquisition**: Single-objective (EI, UCB) and multi-objective (NEHVI). Internal to recommenders.
- **Executor**: Runs experiments. HumanExecutor for manual, ClaudeLightExecutor for autonomous closed-loop

## Executor Interface

Executors bridge Folio's suggestions and actual experiment execution. The base class provides validation and error handling; subclasses implement `_run()`.

```python
class Executor(ABC):
    def execute(self, suggestion: dict, project: Project) -> Observation:
        """Validate inputs and run experiment."""
        # Validates suggestion against project schema, then calls _run()

    @abstractmethod
    def _run(self, suggestion: dict, project: Project) -> Observation:
        """Subclass implements actual experiment execution."""

class HumanExecutor(Executor):
    """Prompts user for actual inputs and outputs via CLI."""

class ClaudeLightExecutor(Executor):
    """Calls Claude-Light API for autonomous closed-loop experiments."""
    def __init__(self, api_url: str = "https://claude-light.cheme.cmu.edu/api"):
        self.api_url = api_url
```

### Using Executors with Folio

```python
from folio.executors import HumanExecutor, ClaudeLightExecutor

# Manual experiments
folio.build_executor("human")
observations = folio.execute("my_project", n_iter=5)

# Autonomous closed-loop
folio.build_executor("claude_light")
observations = folio.execute("my_project", n_iter=20, stop_on_error=False)

# Custom executor
class MyRobotExecutor(Executor):
    def _run(self, suggestion, project):
        # Call robot API...
        return Observation(project_id=project.id, inputs=..., outputs=...)

observations = folio.execute("my_project", n_iter=10, executor=MyRobotExecutor())
```

## Database Configuration

```python
# Local SQLite (default)
db = Database("project.db")

# libSQL cloud sync (for Claude-Light collaboration)
db = Database(
    "project.db",
    sync_url="libsql://your-db.turso.io",
    auth_token="your-token"
)
# Calls conn.sync() after writes
```

## Data Flow

```
User enters observation (inputs, outputs)
        ↓
    Store in SQLite/libSQL
        ↓
    Target.compute(observation) → scalar y per target
        ↓
    Surrogate.fit(X, Y)  # Y shape (n, m) for m objectives
        ↓
    Recommender.recommend() → next inputs
        ↓
    Executor.run(inputs) → outputs  [or display to user]
        ↓
    Loop
```

## Coding Conventions

- Python 3.10+
- Type hints on all public functions
- Use modern syntax: `dict` not `Dict`, `str | None` not `Optional[str]`
- Docstrings: NumPy-style, thorough on both public and internal functions
- Format: black (line length 88)
- Lint: ruff
- Tests: pytest, mirror src/ structure in tests/
- Logging: use `logging` module, never print()
- Errors: raise custom exceptions from exceptions.py

## Style Principles

- Prefer simple over clever
- No unnecessary abstraction
- Functions < 30 lines where possible
- Variable names: descriptive but not verbose
- One blank line between functions, two between classes

## Comments

- All comments on their own line, never trailing
- Comments explain *why*, not *what*
- No obvious comments that restate the code
- Algorithm references go in docstring as "Reference: Author (Year)"

## Docstrings

Write thorough NumPy-style docstrings. Good documentation is not slop. For functions/methods where the useage is not immidiately obvious, include example in the docstring.

```python
# Good
def build(self, model: Model, best_f: float, maximize: bool) -> AcquisitionFunction:
    """Build a BoTorch-compatible acquisition function.

    Parameters
    ----------
    model : Model
        A fitted BoTorch model (e.g., SingleTaskGP).
    best_f : float
        Best observed target value so far.
    maximize : bool
        If True, seek higher values; if False, seek lower values.

    Returns
    -------
    AcquisitionFunction
        BoTorch-compatible acquisition function with forward(X) method.

    Reference: Jones et al. (1998), Efficient Global Optimization.
    """
```

## Anti-Slop Rules

Slop = bad engineering, not verbose documentation.

1. No comments that restate the code
2. No trailing comments
3. Consistent design patterns across modules
4. Proper task decomposition — functions do one thing
5. No over-engineering for hypothetical futures
6. No catching broad `except Exception`
7. No print() for debugging

## Error Handling

- Define custom exceptions in exceptions.py
- Exception messages should tell user what went wrong AND how to fix it
- Never catch broad `except Exception`

```python
# Good
raise ProjectNotFoundError(
    f"No project named '{name}'. Available: {list_projects()}"
)

# Bad
raise Exception("Project not found")
```

## Testing

- Every public function should have at least one test
- Test file mirrors source: src/folio/core/project.py → tests/test_core/test_project.py
- Use pytest fixtures for common setup (database, sample project)
- Test edge cases: empty inputs, invalid bounds, etc.

### Error Message Matching

When testing that exceptions are raised with specific messages, be FLEXIBLE:
- Use multiple alternative keywords with `|` (e.g., `match="negative|non-negative|must be >= 0"`)
- Use case-insensitive matching with `(?i)` (e.g., `match="(?i)invalid"`)
- Match on semantic meaning, not exact wording
- The implementer should have freedom to phrase error messages naturally

```python
# Good - flexible matching
with pytest.raises(ValueError, match="shape|dimension|size"):
    ...

with pytest.raises(ValueError, match="(?i)nan|missing|invalid"):
    ...

with pytest.raises(ValueError, match="negative|non-negative|>= 0|must be positive"):
    ...

# Bad - brittle exact matching
with pytest.raises(ValueError, match="Array shapes must match exactly"):
    ...
```

## Current State

### Completed (MVP Core)

- [x] Project skeleton
- [x] Data layer (Project, Observation, SQLite CRUD)
  - [x] Schema: InputSpec, OutputSpec with validation
  - [x] Observation with validation, timestamp default, tag, notes
  - [x] Project with target_configs (list), reference_point, recommender_config
  - [x] Project.is_multi_objective() for single vs multi-objective detection
  - [x] Database: create/get/delete project, add/get/delete observations
  - [x] Project.get_training_data() extracts (X, y) arrays from observations
- [x] Target interface
  - [x] ScalarTarget ABC, DirectTarget, DerivedTarget, RatioTarget, DifferenceTarget, DistanceTarget, SlopeTarget
- [x] Surrogate interface
  - [x] Surrogate ABC with fit/predict
  - [x] SingleTaskGPSurrogate: BoTorch single-output GP
  - [x] MultiTaskGPSurrogate: BoTorch multi-output GP with ICM kernel
  - [x] TaskStandardize: per-task outcome transform for multi-objective scale balancing
- [x] Acquisition interface (BoTorch-compatible)
  - [x] Single-objective: ExpectedImprovement, UpperConfidenceBound
  - [x] Multi-objective: NEHVI (qLogNoisyExpectedHypervolumeImprovement)
- [x] Recommender interface
  - [x] Recommender ABC with recommend() and recommend_from_data()
  - [x] RandomRecommender: uniform sampling within bounds
  - [x] BayesianRecommender: GP surrogate + acquisition optimization
- [x] Folio API (high-level user interface)
  - [x] Project CRUD: create_project, list_projects, delete_project, get_project
  - [x] Observation CRUD: add_observation, delete_observation, get_observations (with tag filter)
  - [x] Recommendation: suggest() returns list[dict] for batch support
  - [x] Recommender caching: _recommenders dict, _build_recommender, get_recommender
  - [x] Executor support: build_executor(), execute() for automated loops
- [x] libSQL cloud sync support in Database
  - [x] sync_url and auth_token parameters throughout database layer
  - [x] Folio.__init__ accepts sync_url and auth_token
  - [x] conn.sync() called after commits for cloud sync
- [x] Executor interface
  - [x] Executor ABC with execute() and _run()
  - [x] HumanExecutor: interactive CLI prompts for manual experiments
  - [x] ClaudeLightExecutor: autonomous closed-loop via API
- [x] Demos (AI-generated, synthetic data)
  - [x] Quickstart, multi-objective, lab workflow, edge cases
  - [x] Custom surrogates/acquisitions/recommenders demo
  - [x] Quarto report templates (polymer optimization, iridium sensor)
  - [x] Custom target extensions (R2Target, SternVolmerTargets)
  - [x] Executors demo (automated closed-loop optimization)
- [x] CI/CD
  - [x] Cross-platform testing (Ubuntu, macOS, Windows)
  - [x] Pre-commit hooks (black, ruff, nbstripout)

### Not Yet Implemented

- [ ] Add images field to Observation
- [ ] Add procedure, hazards fields to Project
- [ ] GridRecommender
- [ ] ParEGO acquisition function
- [ ] Streamlit UI

## Commands

```bash
pip install -e ".[dev]"    # install in dev mode
pytest                      # run tests
pytest -v                   # verbose
pytest --cov=folio          # with coverage
ruff check .               # lint
ruff check . --fix         # lint and auto-fix
black .                    # format
black --check .            # check format without changing
```

## Debugging Errors

When asked to review error messages or debug issues:
1. **Explain what the error is** — describe the root cause clearly
2. **Ask before showing the fix** — don't immediately provide code; ask "Want me to show the fix?"
3. Let the developer attempt the fix themselves if they prefer

This encourages learning and avoids spoon-feeding solutions.

## Do NOT

- Over-engineer for hypothetical future needs
- Add dependencies without clear justification
- Use print() for debugging
- Catch broad exceptions
- Add trailing comments
- Write comments that restate the code
- Create deep inheritance hierarchies
- Use abbreviations in variable names (except standard: i, j, n, X, y)
