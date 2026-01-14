"""Human-in-the-loop executor for manual experiment entry."""

from collections.abc import Callable

from folio.core.observation import Observation
from folio.core.project import Project
from folio.exceptions import ExecutorError
from folio.executors.base import Executor


class HumanExecutor(Executor):
    """Executor that prompts a human to run experiments and enter results.

    Displays suggested inputs to the user and collects output values via
    interactive prompts. Supports marking experiments as failed.

    Parameters
    ----------
    input_func : Callable[[str], str]
        Function for collecting user input. Defaults to builtin `input()`.
        Override for testing or custom UI integration.

    Examples
    --------
    >>> executor = HumanExecutor()
    >>> obs = executor.execute({"temperature": 80.0}, project)
    # User is prompted for outputs interactively
    """

    def __init__(self, input_func: Callable[[str], str] = input):
        self.input_func = input_func

    def _run(self, suggestion: dict, project: Project) -> Observation:
        """Prompt user to run experiment and enter results.

        Displays the suggested inputs, then prompts for each output value.
        Also asks if the experiment failed.

        Parameters
        ----------
        suggestion : dict
            Validated input values to display to user.
        project : Project
            Project defining expected outputs.

        Returns
        -------
        Observation
            Observation with user-entered outputs and failed flag.
        """
        print(f"Suggested inputs: {suggestion}")
        print("(Enter 'quit' at any input or output prompt to stop)")

        actual_input = {}
        output = {}

        for name, suggested_val in suggestion.items():
            while True:
                raw = self.input_func(
                    f"Enter actual {name} (suggested: {suggested_val}): "
                ).strip()
                if raw == "":
                    # user just hit enter, yell at them
                    print("You always need to record the actual inputs")
                    continue
                elif raw == "quit":
                    raise ExecutorError("quitted")
                else:
                    try:
                        actual_input[name] = float(raw)
                    except ValueError:
                        print(f"Invalid number: '{raw}'.")
                        continue
                    break

        for output_spec in project.outputs:
            while True:
                raw = self.input_func(f"Enter {output_spec.name}: ").strip()
                if raw == "":
                    # user just hit enter, yell at them
                    print("You always need to record the actual outputs")
                    continue
                elif raw == "quit":
                    raise ExecutorError("quitted")
                else:
                    try:
                        output[output_spec.name] = float(raw)
                    except ValueError:
                        print(f"Invalid number: '{raw}'.")
                        continue
                    break

        failed = (
            self.input_func(
                "Did experiment fail (fail meaning the "
                "output value is not to be trusted)? [y/n]:"
            )
            .lower()
            .startswith("y")
        )

        note = self.input_func("Enter notes/obsevaions for this run: ")
        note = note.strip() if note.strip() else None

        tag = self.input_func("Enter tag for this run: ")
        tag = tag.strip() if tag.strip() else None

        project_id = project.id

        obs = Observation(
            project_id=project_id,
            inputs=actual_input,
            inputs_suggested=suggestion,
            outputs=output,
            notes=note,
            tag=tag,
            failed=failed,
        )

        return obs
