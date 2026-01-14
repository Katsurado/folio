from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from folio.core.database import (
    add_observation,
    create_project,
    delete_observation,
    delete_project,
    get_connection,
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
        target_configs=[TargetConfig(objective="yield", objective_mode="maximize")],
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
        assert retrieved.target_configs[0].objective == "yield"
        assert retrieved.target_configs[0].objective_mode == "maximize"

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


class TestLibSQLConnection:
    """Tests for libSQL cloud sync functionality using mocked connections."""

    @pytest.fixture
    def mock_libsql_conn(self):
        """Create a mock libsql connection with all required methods."""
        conn = MagicMock()
        conn.row_factory = None
        conn.execute.return_value = MagicMock()
        conn.executescript.return_value = None
        conn.commit.return_value = None
        conn.rollback.return_value = None
        conn.sync.return_value = None
        conn.close.return_value = None
        return conn

    def test_get_connection_uses_libsql_when_sync_url_provided(
        self, temp_db, mock_libsql_conn
    ):
        """Verify libsql.connect is called when sync_url is provided."""
        with patch("folio.core.database.libsql") as mock_libsql:
            mock_libsql.connect.return_value = mock_libsql_conn

            with get_connection(
                temp_db,
                sync_url="libsql://test.turso.io",
                auth_token="test-token",
            ):
                pass

            mock_libsql.connect.assert_called_once_with(
                str(temp_db),
                sync_url="libsql://test.turso.io",
                auth_token="test-token",
            )

    def test_get_connection_calls_sync_after_commit_for_libsql(
        self, temp_db, mock_libsql_conn
    ):
        """Verify conn.sync() is called after commit for libsql connections."""
        with patch("folio.core.database.libsql") as mock_libsql:
            mock_libsql.connect.return_value = mock_libsql_conn

            with get_connection(
                temp_db,
                sync_url="libsql://test.turso.io",
                auth_token="test-token",
            ) as conn:
                conn.execute("SELECT 1")

            mock_libsql_conn.commit.assert_called_once()
            mock_libsql_conn.sync.assert_called_once()

    def test_get_connection_no_sync_for_sqlite(self, temp_db):
        """Verify sync is not called for regular SQLite connections."""
        with patch("folio.core.database.sqlite3") as mock_sqlite3:
            mock_conn = MagicMock()
            mock_sqlite3.connect.return_value = mock_conn

            with get_connection(temp_db) as conn:
                conn.execute("SELECT 1")

            mock_conn.commit.assert_called_once()
            mock_conn.sync.assert_not_called()

    def test_init_db_with_libsql_calls_sync(self, temp_db, mock_libsql_conn):
        """Verify init_db with libsql calls sync after schema creation."""
        with patch("folio.core.database.libsql") as mock_libsql:
            mock_libsql.connect.return_value = mock_libsql_conn

            init_db(
                temp_db,
                sync_url="libsql://test.turso.io",
                auth_token="test-token",
            )

            mock_libsql.connect.assert_called_once()
            mock_libsql_conn.executescript.assert_called_once()
            mock_libsql_conn.commit.assert_called_once()
            mock_libsql_conn.sync.assert_called_once()
            mock_libsql_conn.close.assert_called_once()

    def test_create_project_with_libsql_syncs(self, temp_db, sample_project):
        """Verify create_project syncs when using libsql."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 1
        mock_conn.execute.return_value = mock_cursor

        with patch("folio.core.database.libsql") as mock_libsql:
            mock_libsql.connect.return_value = mock_conn

            create_project(
                sample_project,
                db_path=temp_db,
                sync_url="libsql://test.turso.io",
                auth_token="test-token",
            )

            # Should sync twice: once for init_db and once for create_project
            assert mock_conn.sync.call_count == 2

    def test_add_observation_with_libsql_syncs(self, temp_db, sample_project):
        """Verify add_observation syncs when using libsql."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 1
        mock_conn.execute.return_value = mock_cursor

        with patch("folio.core.database.libsql") as mock_libsql:
            mock_libsql.connect.return_value = mock_conn

            # First create a project
            create_project(
                sample_project,
                db_path=temp_db,
                sync_url="libsql://test.turso.io",
                auth_token="test-token",
            )

            # Reset sync call count
            mock_conn.sync.reset_mock()

            obs = Observation(
                project_id=1,
                inputs={"temperature": 50.0, "solvent": "water"},
                outputs={"yield": 85.5},
            )
            add_observation(
                obs,
                db_path=temp_db,
                sync_url="libsql://test.turso.io",
                auth_token="test-token",
            )

            mock_conn.sync.assert_called_once()

    def test_delete_project_with_libsql_syncs(self, temp_db):
        """Verify delete_project syncs when using libsql."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_conn.execute.return_value = mock_cursor

        with patch("folio.core.database.libsql") as mock_libsql:
            mock_libsql.connect.return_value = mock_conn

            delete_project(
                "test_project",
                db_path=temp_db,
                sync_url="libsql://test.turso.io",
                auth_token="test-token",
            )

            mock_conn.sync.assert_called_once()

    def test_delete_observation_with_libsql_syncs(self, temp_db):
        """Verify delete_observation syncs when using libsql."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_conn.execute.return_value = mock_cursor

        with patch("folio.core.database.libsql") as mock_libsql:
            mock_libsql.connect.return_value = mock_conn

            delete_observation(
                observation_id=1,
                db_path=temp_db,
                sync_url="libsql://test.turso.io",
                auth_token="test-token",
            )

            mock_conn.sync.assert_called_once()

    def test_get_connection_rollback_on_exception(self, temp_db, mock_libsql_conn):
        """Verify connection rolls back on exception."""
        with patch("folio.core.database.libsql") as mock_libsql:
            mock_libsql.connect.return_value = mock_libsql_conn

            with pytest.raises(ValueError):
                with get_connection(
                    temp_db,
                    sync_url="libsql://test.turso.io",
                    auth_token="test-token",
                ):
                    raise ValueError("Test error")

            mock_libsql_conn.rollback.assert_called_once()
            mock_libsql_conn.sync.assert_not_called()

    def test_get_connection_closes_on_exception(self, temp_db, mock_libsql_conn):
        """Verify connection is closed even on exception."""
        with patch("folio.core.database.libsql") as mock_libsql:
            mock_libsql.connect.return_value = mock_libsql_conn

            with pytest.raises(ValueError):
                with get_connection(
                    temp_db,
                    sync_url="libsql://test.turso.io",
                    auth_token="test-token",
                ):
                    raise ValueError("Test error")

            mock_libsql_conn.close.assert_called_once()
