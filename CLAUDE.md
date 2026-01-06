# CLAUDE.md

## Project Overview

Folio is an open-source electronic lab notebook with intelligent experiment suggestions for chemistry labs. Human-in-the-loop MVP, with architecture enabling closed-loop automation later.

Target users: lab scientists who know some Python but aren't software engineers.

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

- **Project**: Experiment schema (inputs, outputs, target config, recommender config)
- **Observation**: Single data point (inputs dict, outputs dict, timestamp, notes, tag, raw_data_path)
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
- Docstrings: Google style, one line if possible
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

- One line if possible, elaborate only if non-obvious
- Never start with "This function..."
- Don't repeat type hints in prose
- For algorithms: name it and cite, don't explain the math in docstring

```python
# Good
def expected_improvement(X: np.ndarray, surrogate: Surrogate, best_y: float) -> np.ndarray:
    """Compute Expected Improvement scores for candidate points.

    Reference: Jones et al. (1998), Efficient Global Optimization.
    """

# Bad
def expected_improvement(X, surrogate, best_y):
    """
    This function computes the Expected Improvement acquisition function.

    Expected Improvement is defined as EI(x) = E[max(f(x) - f(x*), 0)]...
    [200 more words]

    Args:
        X: The candidate points to evaluate (this is a numpy array).
    """
```

## Anti-Slop Rules

1. Docstrings: one line if possible
2. No "This function..." or "This method..."
3. Don't repeat type hints in docstring prose
4. No comments that restate the code
5. No trailing comments, ever
6. If a function needs 10+ lines of docstring, the function is too complex
7. No unnecessary blank lines
8. No over-engineering for hypothetical futures

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
- [ ] Data layer (Project, Observation, SQLite CRUD)
- [ ] Target interface
- [ ] Recommender interface
- [ ] Surrogate interface + GP implementation
- [ ] Acquisition interface + EI/UCB implementation
- [ ] BORecommender
- [ ] RandomRecommender, GridRecommender
- [ ] Executor interface + HumanExecutor
- [ ] CLI/API (create_project, record, suggest_next, to_dataframe)
- [ ] Export to qmd
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
- Write verbose docstrings that repeat type hints
- Use print() for debugging
- Catch broad exceptions
- Add trailing comments
- Write obvious comments
- Create deep inheritance hierarchies
- Use abbreviations in variable names (except standard: i, j, n, X, y)
