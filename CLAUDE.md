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
│   └── multitask_gp.py # MultiTaskGPSurrogate (BoTorch, multi-output GP with ICM)
├── targets/
│   ├── base.py         # ScalarTarget ABC
│   └── builtin.py      # DirectTarget, RatioTarget, etc.
├── executors/          # Executor interface (HumanExecutor, ClaudeLightExecutor)
├── ui/                 # Streamlit app
├── api.py              # High-level user-facing functions
└── exceptions.py       # Custom exceptions
```

Note: `targets/` uses `TYPE_CHECKING` for `Observation` imports to avoid circular dependencies with `core/`.

## Key Abstractions

- **Project**: Experiment schema (inputs, outputs, target config, recommender config, procedure, hazards)
- **Observation**: Single data point (inputs dict, outputs dict, timestamp, notes, tag, images, failed)
- **Target**: Extracts scalar optimization target from Observation (direct or derived from outputs)
- **Recommender**: Suggests next experiments. Interface: recommend(observations) → dict, recommend_from_data(X, y, bounds, objective) → np.ndarray. Implementations: BayesianRecommender, RandomRecommender
- **Surrogate**: Model that fits observations. Interface: fit(X, y), predict(X) → (mean, std). SingleTaskGPSurrogate for scalar targets, MultiTaskGPSurrogate for correlated multi-output targets.
- **Acquisition**: Builds BoTorch-compatible acquisition functions (internal to recommenders). Interface: build(model, best_f, maximize) → AcquisitionFunction. The returned function has forward(X) for use with optimize_acqf.
- **Executor**: Runs experiments. HumanExecutor for manual, ClaudeLightExecutor for autonomous closed-loop

## Executor Interface

```python
class Executor(ABC):
    @abstractmethod
    def run(self, inputs: dict) -> dict:
        """Run experiment, return outputs."""

class HumanExecutor(Executor):
    def run(self, inputs):
        # Display to user, wait for manual entry

class ClaudeLightExecutor(Executor):
    def __init__(self, api_url):
        self.api_url = api_url

    def run(self, inputs):
        res = requests.get(self.api_url, params=inputs)
        return res.json()["out"]
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
    Target.compute(observation) → scalar y
        ↓
    Surrogate.fit(X, y)
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

- [x] Project skeleton
- [x] Data layer (Project, Observation, SQLite CRUD)
  - [x] Schema: InputSpec, OutputSpec, ConstantSpec with validation
  - [x] Observation with validation, timestamp default, failed flag
  - [x] Project with TargetConfig, RecommenderConfig, validation
  - [x] Database: create/get/delete project, add/get observations
  - [x] Project.get_training_data() extracts X, y arrays from observations
- [x] Target interface
  - [x] ScalarTarget abstract base class (folio/targets/base.py)
  - [x] DirectTarget: extracts single output value
  - [x] DerivedTarget: computes via custom function
  - [x] RatioTarget: numerator / denominator
  - [x] DifferenceTarget: first - second
  - [x] DistanceTarget: euclidean/mse/mae distance to target values
  - [x] SlopeTarget: linear fit slope across multiple outputs
- [x] Surrogate interface
  - [x] Surrogate ABC with fit/predict (folio/surrogates/base.py)
  - [x] SingleTaskGPSurrogate: BoTorch-based single-output GP (Matérn 2.5, ARD, learned noise)
  - [x] MultiTaskGPSurrogate: BoTorch-based multi-output GP with ICM kernel for correlated outputs
- [x] Acquisition interface (BoTorch-compatible)
  - [x] Acquisition ABC with build(model, best_f, maximize) → AcquisitionFunction
  - [x] ExpectedImprovement: EI with xi parameter, returns _EIAcquisition
  - [x] UpperConfidenceBound: UCB with beta parameter, returns _UCBAcquisition
  - [x] Inner classes implement forward(X) for use with optimize_acqf
- [x] Recommender interface (prototypes with tests, awaiting implementation)
  - [x] Recommender ABC with recommend(observations) and recommend_from_data(X, y, bounds, objective)
  - [x] BayesianRecommender: GP surrogate + acquisition optimization (prototype)
  - [x] RandomRecommender: uniform sampling within bounds (prototype)
  - [x] Project.get_optimization_bounds() returns (2, d) array for BoTorch
- [ ] Add images field to Observation
- [ ] Add procedure, hazards fields to Project
- [ ] libSQL cloud sync support in Database
- [ ] GridRecommender
- [ ] Executor interface
  - [ ] HumanExecutor
  - [ ] ClaudeLightExecutor
- [ ] Export to PDF via Quarto
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
