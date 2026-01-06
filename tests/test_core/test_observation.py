from datetime import datetime

import pytest

from folio.core.observation import Observation
from folio.exceptions import InvalidInputError, InvalidOutputError


class TestObservation:
    def test_valid_observation_all_fields(self):
        ts = datetime(2024, 1, 15, 10, 30, 0)
        obs = Observation(
            project_id=1,
            inputs={"temperature": 50.0, "solvent": "water"},
            outputs={"yield": 85.5},
            timestamp=ts,
            id=42,
            notes="First run",
            tag="batch_1",
            raw_data_path="/data/exp_001.csv",
        )
        assert obs.project_id == 1
        assert obs.inputs == {"temperature": 50.0, "solvent": "water"}
        assert obs.outputs == {"yield": 85.5}
        assert obs.timestamp == ts
        assert obs.id == 42
        assert obs.notes == "First run"
        assert obs.tag == "batch_1"
        assert obs.raw_data_path == "/data/exp_001.csv"

    def test_valid_observation_optional_fields_none(self):
        ts = datetime(2024, 1, 15, 10, 30, 0)
        obs = Observation(
            project_id=1,
            inputs={"temperature": 50.0},
            outputs={"yield": 85.5},
            timestamp=ts,
        )
        assert obs.project_id == 1
        assert obs.inputs == {"temperature": 50.0}
        assert obs.outputs == {"yield": 85.5}
        assert obs.timestamp == ts
        assert obs.id is None
        assert obs.notes is None
        assert obs.tag is None
        assert obs.raw_data_path is None

    def test_timestamp_defaults_to_now(self):
        before = datetime.now()
        obs = Observation(
            project_id=1,
            inputs={"temperature": 50.0},
            outputs={"yield": 85.5},
        )
        after = datetime.now()
        assert before <= obs.timestamp <= after

    def test_timestamp_preserves_provided_value(self):
        ts = datetime(2020, 6, 15, 12, 0, 0)
        obs = Observation(
            project_id=1,
            inputs={"temperature": 50.0},
            outputs={"yield": 85.5},
            timestamp=ts,
        )
        assert obs.timestamp == ts

    def test_invalid_project_id_zero_raises(self):
        with pytest.raises(InvalidInputError, match="positive integer"):
            Observation(
                project_id=0,
                inputs={"temperature": 50.0},
                outputs={"yield": 85.5},
            )

    def test_invalid_project_id_negative_raises(self):
        with pytest.raises(InvalidInputError, match="positive integer"):
            Observation(
                project_id=-1,
                inputs={"temperature": 50.0},
                outputs={"yield": 85.5},
            )

    def test_invalid_project_id_wrong_type_raises(self):
        with pytest.raises(InvalidInputError, match="positive integer"):
            Observation(
                project_id="1",
                inputs={"temperature": 50.0},
                outputs={"yield": 85.5},
            )

    def test_invalid_inputs_wrong_type_raises(self):
        with pytest.raises(InvalidInputError, match="inputs must be a dict"):
            Observation(
                project_id=1,
                inputs=[("temperature", 50.0)],
                outputs={"yield": 85.5},
            )

    def test_invalid_outputs_wrong_type_raises(self):
        with pytest.raises(InvalidOutputError, match="outputs must be a dict"):
            Observation(
                project_id=1,
                inputs={"temperature": 50.0},
                outputs=[("yield", 85.5)],
            )

    def test_invalid_timestamp_wrong_type_raises(self):
        with pytest.raises(InvalidInputError, match="timestamp must be a datetime"):
            Observation(
                project_id=1,
                inputs={"temperature": 50.0},
                outputs={"yield": 85.5},
                timestamp="2024-01-15",
            )

    def test_failed_defaults_to_false(self):
        obs = Observation(
            project_id=1,
            inputs={"temperature": 50.0},
            outputs={"yield": 85.5},
        )
        assert obs.failed is False

    def test_failed_can_be_set_true(self):
        obs = Observation(
            project_id=1,
            inputs={"temperature": 50.0},
            outputs={"yield": 85.5},
            failed=True,
        )
        assert obs.failed is True
