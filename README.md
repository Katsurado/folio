# Folio

An open-source electronic lab notebook with intelligent experiment suggestions.

STILL BEING DEVELOPED, COMPLETE TIME TBD

## What is Folio?

Folio helps lab scientists run experiments more efficiently. You record your experimental results, and Folio suggests what to try next using Bayesian optimization (or other methods you choose).

```
Record results → Folio suggests next experiment → Run it → Repeat
```

## Features

- **Simple data entry**: GUI or Python API
- **Smart suggestions**: Bayesian optimization to find optimal conditions faster
- **Flexible**: Define your own inputs, outputs, and optimization targets
- **Extensible**: Plug in custom models and acquisition functions
- **Export**: Generate Quarto documents for analysis and PDF reports

## Installation

```bash
# From GitHub
pip install git+https://github.com/Katsurado/folio.git

# For development
git clone https://github.com/Katsurado/folio.git
cd folio
pip install -e ".[dev]"
```

## Quick Start

```python
from folio import create_project, record, suggest_next

# Define your experiment
project = create_project(
    name="suzuki_coupling",
    inputs=[
        {"name": "temperature", "type": "continuous", "bounds": [60, 120]},
        {"name": "catalyst_loading", "type": "continuous", "bounds": [0.01, 0.10]},
        {"name": "solvent", "type": "categorical", "levels": ["THF", "DMF", "toluene"]},
    ],
    outputs=[
        {"name": "yield"},
    ],
    target="yield",
    objective="maximize",
)

# Record an experiment
record(
    project="suzuki_coupling",
    inputs={"temperature": 80, "catalyst_loading": 0.05, "solvent": "THF"},
    outputs={"yield": 72.5},
)

# Get suggestion for next experiment
next_exp = suggest_next("suzuki_coupling")
print(next_exp)
# {"temperature": 95.2, "catalyst_loading": 0.03, "solvent": "DMF"}
```

## Documentation

See [docs/](docs/) for full documentation.

## License

MIT

## Statement on AI
Code in this project was written with assistance from generative AI models such as Claude. I have
a working knowledge of programming and basic SWE practices, but I am a chemist first. AI assistance
help briges the gap between "functional code in research labs" and "well-architechted code that is
readable and maintainable", which is crucial for Folio to be a actually usable tool.

All design decision and validation remain my own.
