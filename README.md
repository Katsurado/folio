# Folio

An open-source electronic lab notebook with intelligent experiment suggestions for chemistry labs.

## What is Folio?

Folio helps lab scientists run experiments more efficiently. You record your experimental results, and Folio suggests what to try next using Bayesian optimization.

```
Record results → Folio suggests next experiment → Run it → Repeat
```

## Features

- **Simple data entry**: Python API (Streamlit GUI coming soon)
- **Smart suggestions**: Bayesian optimization (single and multi-objective)
- **Closed-loop automation**: Executors for human-in-the-loop or fully autonomous experiments
- **Cloud sync**: libSQL support for distributed collaboration
- **Flexible**: Define your own inputs, outputs, and optimization targets
- **Extensible**: Plug in custom surrogates, acquisition functions, recommenders, and executors

## Installation

```bash
# From GitHub
pip install git+https://github.com/Katsurado/folio.git

# For development
git clone https://github.com/Katsurado/folio.git
cd folio
pip install -e ".[dev]"

# For running demos (includes Jupyter, Quarto support)
pip install -e ".[demos]"
```

## Quick Start

```python
from folio.api import Folio
from folio.core.schema import InputSpec, OutputSpec
from folio.core.config import TargetConfig

# Create Folio instance
folio = Folio("my_experiments.db")

# Define your experiment
folio.create_project(
    name="suzuki_coupling",
    inputs=[
        InputSpec("temperature", "continuous", bounds=(60, 120), units="°C"),
        InputSpec("catalyst_loading", "continuous", bounds=(0.01, 0.10), units="mol%"),
        InputSpec("solvent", "categorical", levels=["THF", "DMF", "toluene"]),
    ],
    outputs=[OutputSpec("yield", units="%")],
    target_configs=[TargetConfig(objective="yield", objective_mode="maximize")],
)

# Record an experiment
folio.add_observation(
    project_name="suzuki_coupling",
    inputs={"temperature": 80, "catalyst_loading": 0.05, "solvent": "THF"},
    outputs={"yield": 72.5},
)

# Get suggestion for next experiment
suggestions = folio.suggest("suzuki_coupling")
print(suggestions[0])
# {"temperature": 95.2, "catalyst_loading": 0.03, "solvent": "DMF"}
```

## Automated Optimization

Use executors to automate the experiment loop:

```python
from folio.executors import Executor
from folio.core.observation import Observation

# Create a custom executor (e.g., for simulation or robot API)
class SimulatorExecutor(Executor):
    def _run(self, suggestion, project):
        # Run your experiment here
        outputs = {"yield": simulate_reaction(**suggestion)}
        return Observation(
            project_id=project.id,
            inputs=suggestion,
            outputs=outputs,
        )

# Run 20 automated iterations
observations = folio.execute(
    project_name="suzuki_coupling",
    n_iter=20,
    executor=SimulatorExecutor(),
)
```

Built-in executors:
- `HumanExecutor`: Interactive CLI prompts for manual experiments
- `ClaudeLightExecutor`: Fully autonomous via Claude-Light API

## Demos

See [`demos/`](demos/) for example notebooks and Quarto reports demonstrating Folio usage.

**Note:** Demos are AI-generated and contain synthetic data. The scientific content has not been verified and should not be used for actual research.

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
