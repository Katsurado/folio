# CLAUDE.md

## Project Overview

Folio is an open-source electronic lab notebook with intelligent experiment suggestions for chemistry labs. Human-in-the-loop MVP, with architecture enabling closed-loop automation later.

Target users: lab scientists. No coding required (GUI), but API available for those who want it.

## MVP Scope

Three pillars, all required:
1. **ELN**: Create projects, record observations (CRUD), attach images, document procedures/hazards
2. **PDF Export**: Quarto-based, professional lab notebook output
3. **BO Suggestions**: Smart next-experiment recommendations

Two interfaces:
- **Streamlit GUI**: Full workflow for non-coders
- **Python API**: Same functionality for Jupyter/Quarto power users

Plus:
- **Documentation**: Proper docstrings (NumPy-style) on all public and internal functions

## Architecture

```
src/folio/
├── core/           # Project, Observation, database
├── recommenders/   # Recommender interface + implementations (BO, random, grid)
├── surrogates/     # Surrogate interface + implementations (GP, etc.)
├── acquisitions/   # Acquisition interface + implementations (EI, UCB)
├── targets/        # Target interface (direct, derived)
├── executors/      # Executor interface (human, instrument)
├── ui/             # Streamlit app
├── api.py          # High-level user-facing functions
└── exceptions.py   # Custom exceptions
```

## Key Abstractions

- **Project**: Experiment schema (inputs, outputs, target config, recommender config, procedure, hazards)
- **Observation**: Single data point (inputs dict, outputs dict, timestamp, notes, tag, images, failed)
- **Target**: Extracts scalar optimization target from Observation (direct or derived from outputs)
- **Recommender**: Suggests next experiments. Implementations: BORecommender, RandomRecommender, GridRecommender
- **Surrogate**: Model that fits observations. Interface: fit(X, y), predict(X) → (mean, std)
- **Acquisition**: Scores candidate points. Interface: evaluate(X, surrogate, best_y) → scores
- **Executor**: Runs experiments. HumanExecutor for MVP, InstrumentExecutor for future closed-loop

## Data Flow

```
User enters observation (inputs, outputs)
        ↓
    Store in SQLite
        ↓
    Target.compute(observation) → scalar y
        ↓
    Surrogate.fit(X, y)
        ↓
    Recommender.suggest() → next inputs
        ↓
    Display to user
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

Write thorough NumPy-style docstrings. Good documentation is not slop.

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
- [ ] Recommender interface
- [ ] Surrogate interface + GP implementation
- [ ] Acquisition interface + EI/UCB implementation
- [ ] BORecommender
- [ ] RandomRecommender, GridRecommender
- [ ] Executor interface + HumanExecutor
- [ ] Export to PDF via Quarto
- [ ] Streamlit UI
- [ ] Add images field to Observation
- [ ] Add procedure, hazards fields to Project
- [ ] Proper docstrings on all functions (NumPy-style)

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
