# CLAUDE.md

## Project Overview

Folio is an open-source electronic lab notebook with intelligent experiment suggestions for chemistry labs. Human-in-the-loop MVP, with architecture enabling closed-loop automation.

Target users: lab scientists. No coding required (GUI), but API available for those who want it.

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
├── recommenders/       # Recommender interface + implementations (BO, random, grid)
│   └── acquisitions/   # Acquisition functions (EI, UCB) - internal to recommenders
├── surrogates/
│   ├── base.py         # Surrogate ABC
│   └── gp.py           # SingleTaskGPSurrogate (BoTorch, single-output GP)
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
- **Recommender**: Suggests next experiments. Implementations: BORecommender, RandomRecommender, GridRecommender
- **Surrogate**: Model that fits observations. Interface: fit(X, y), predict(X) → (mean, std). SingleTaskGPSurrogate for scalar targets.
- **Acquisition**: Scores candidate points (internal to recommenders). Interface: evaluate(X, surrogate, best_y) → scores
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
    Recommender.suggest() → next inputs
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
def expected_improvement(X: np.ndarray, surrogate: Surrogate, best_y: float) -> np.ndarray:
    """Compute Expected Improvement scores for candidate points.

    Parameters
    ----------
    X : np.ndarray, shape (n_candidates, n_features)
        Candidate points to evaluate.
    surrogate : Surrogate
        Fitted surrogate model with predict(X) -> (mean, std).
    best_y : float
        Best observed target value so far.

    Returns
    -------
    np.ndarray, shape (n_candidates,)
        EI score for each candidate point.

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
- [ ] Add images field to Observation
- [ ] Add procedure, hazards fields to Project
- [ ] libSQL cloud sync support in Database
- [ ] Recommender interface
- [ ] Acquisition interface + EI/UCB implementation
- [ ] BORecommender
- [ ] RandomRecommender, GridRecommender
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

## Do NOT

- Over-engineer for hypothetical future needs
- Add dependencies without clear justification
- Use print() for debugging
- Catch broad exceptions
- Add trailing comments
- Write comments that restate the code
- Create deep inheritance hierarchies
- Use abbreviations in variable names (except standard: i, j, n, X, y)
