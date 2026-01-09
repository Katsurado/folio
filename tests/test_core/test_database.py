from datetime import datetime

import pytest

from folio.core.database import (
    add_observation,
    create_project,
    delete_project,
    get_observations,
    get_project,
    init_db,
)
from folio.core.observation import Observation
from folio.core.project import Project, TargetConfig
from folio.core.schema import InputSpec, OutputSpec
from folio.exceptions import ProjectExistsError, ProjectNotFoundError


@pytest.fixture
def sample_project():
    """Create a sample project for testing."""
    return Project(
        id=None,
        name="test_project",
        inputs=[
            InputSpec(name="temperature", type="continuous", bounds=(20.0, 100.0)),
            InputSpec(name="solvent", type="categorical", levels=["water", "ethanol"]),
        ],
        outputs=[OutputSpec(name="yield", units="%")],
        target_config=TargetConfig(objective="yield", objective_mode="maximize"),
    )


class TestProjectCRUD:
    def test_save_and_retrieve_project(self, temp_db, sample_project):
        created = create_project(sample_project, db_path=temp_db)
        assert created.id is not None

        retrieved = get_project("test_project", db_path=temp_db)
        assert retrieved.id == created.id
        assert retrieved.name == "test_project"
        assert len(retrieved.inputs) == 2
        assert retrieved.inputs[0].name == "temperature"
        # JSON serialization converts tuples to lists
        assert list(retrieved.inputs[0].bounds) == [20.0, 100.0]
        assert retrieved.inputs[1].name == "solvent"
        assert retrieved.inputs[1].levels == ["water", "ethanol"]
        assert len(retrieved.outputs) == 1
        assert retrieved.outputs[0].name == "yield"
        assert retrieved.outputs[0].units == "%"
        assert retrieved.target_config.objective == "yield"
        assert retrieved.target_config.objective_mode == "maximize"

    def test_delete_project(self, temp_db, sample_project):
        create_project(sample_project, db_path=temp_db)
        delete_project("test_project", db_path=temp_db)

        with pytest.raises(ProjectNotFoundError):
            get_project("test_project", db_path=temp_db)

    def test_get_missing_project_raises(self, temp_db):
        with pytest.raises(ProjectNotFoundError, match="No project named"):
            get_project("nonexistent", db_path=temp_db)

    def test_delete_missing_project_raises(self, temp_db):
        init_db(temp_db)
        with pytest.raises(ProjectNotFoundError, match="No project named"):
            delete_project("nonexistent", db_path=temp_db)

    def test_duplicate_project_raises(self, temp_db, sample_project):
        create_project(sample_project, db_path=temp_db)

        with pytest.raises(ProjectExistsError, match="already exists"):
            create_project(sample_project, db_path=temp_db)

    def test_project_id_none_before_save_assigned_after(self, temp_db, sample_project):
        assert sample_project.id is None
        created = create_project(sample_project, db_path=temp_db)
        assert isinstance(created.id, int)
        assert created.id >= 1


class TestObservationCRUD:
    def test_save_and_retrieve_observation(self, temp_db, sample_project):
        created_project = create_project(sample_project, db_path=temp_db)
        ts = datetime(2024, 1, 15, 10, 30, 0)

        obs = Observation(
            project_id=created_project.id,
            inputs={"temperature": 50.0, "solvent": "water"},
            outputs={"yield": 85.5},
            timestamp=ts,
            notes="First experiment",
        )
        added = add_observation(obs, db_path=temp_db)
        assert added.id is not None

        observations = get_observations(created_project.id, db_path=temp_db)
        assert len(observations) == 1
        assert observations[0].id == added.id
        assert observations[0].project_id == created_project.id
        assert observations[0].inputs == {"temperature": 50.0, "solvent": "water"}
        assert observations[0].outputs == {"yield": 85.5}
        assert observations[0].timestamp.year == 2024
        assert observations[0].timestamp.month == 1
        assert observations[0].timestamp.day == 15
        assert observations[0].timestamp.hour == 10
        assert observations[0].timestamp.minute == 30
        assert observations[0].notes == "First experiment"

    def test_get_observations_empty_list(self, temp_db, sample_project):
        created_project = create_project(sample_project, db_path=temp_db)

        observations = get_observations(created_project.id, db_path=temp_db)
        assert observations == []

    def test_get_observations_multiple(self, temp_db, sample_project):
        created_project = create_project(sample_project, db_path=temp_db)

        obs1 = Observation(
            project_id=created_project.id,
            inputs={"temperature": 50.0, "solvent": "water"},
            outputs={"yield": 80.0},
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
        )
        obs2 = Observation(
            project_id=created_project.id,
            inputs={"temperature": 75.0, "solvent": "ethanol"},
            outputs={"yield": 90.0},
            timestamp=datetime(2024, 1, 15, 11, 0, 0),
        )
        add_observation(obs1, db_path=temp_db)
        add_observation(obs2, db_path=temp_db)

        observations = get_observations(created_project.id, db_path=temp_db)
        assert len(observations) == 2
        # Should be ordered by timestamp
        assert observations[0].inputs["temperature"] == 50.0
        assert observations[1].inputs["temperature"] == 75.0

    def test_delete_project_cascades_observations(self, temp_db, sample_project):
        created_project = create_project(sample_project, db_path=temp_db)
        obs = Observation(
            project_id=created_project.id,
            inputs={"temperature": 50.0, "solvent": "water"},
            outputs={"yield": 85.5},
        )
        add_observation(obs, db_path=temp_db)

        delete_project("test_project", db_path=temp_db)

        # Observations should be deleted with project
        observations = get_observations(created_project.id, db_path=temp_db)
        assert observations == []

    def test_observation_id_none_before_save_assigned_after(
        self, temp_db, sample_project
    ):
        created_project = create_project(sample_project, db_path=temp_db)
        obs = Observation(
            project_id=created_project.id,
            inputs={"temperature": 50.0, "solvent": "water"},
            outputs={"yield": 85.5},
        )
        assert obs.id is None
        added = add_observation(obs, db_path=temp_db)
        assert isinstance(added.id, int)
        assert added.id >= 1

    def test_observation_ids_sequential(self, temp_db, sample_project):
        created_project = create_project(sample_project, db_path=temp_db)
        obs1 = Observation(
            project_id=created_project.id,
            inputs={"temperature": 50.0, "solvent": "water"},
            outputs={"yield": 80.0},
        )
        obs2 = Observation(
            project_id=created_project.id,
            inputs={"temperature": 75.0, "solvent": "ethanol"},
            outputs={"yield": 90.0},
        )
        obs3 = Observation(
            project_id=created_project.id,
            inputs={"temperature": 60.0, "solvent": "water"},
            outputs={"yield": 85.0},
        )
        added1 = add_observation(obs1, db_path=temp_db)
        added2 = add_observation(obs2, db_path=temp_db)
        added3 = add_observation(obs3, db_path=temp_db)

        assert added2.id == added1.id + 1
        assert added3.id == added2.id + 1

    def test_observation_nonexistent_project_raises(self, temp_db):
        init_db(temp_db)
        obs = Observation(
            project_id=9999,
            inputs={"temperature": 50.0, "solvent": "water"},
            outputs={"yield": 85.5},
        )
        with pytest.raises(Exception):
            add_observation(obs, db_path=temp_db)
