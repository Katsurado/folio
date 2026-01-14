"""SQLite database operations for projects and observations.

Supports both local SQLite and cloud libSQL backends. For cloud sync,
provide sync_url and auth_token parameters to get_connection().

Environment variables for libSQL cloud sync:
    CLAUDELIGHT_DB_URL: libSQL database URL (e.g., libsql://your-db.turso.io)
    CLAUDELIGHT_RW: Read-write authentication token
    CLAUDELIGHT_RO: Read-only authentication token
"""

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import libsql

from folio.core.config import RecommenderConfig, TargetConfig
from folio.core.observation import Observation
from folio.core.project import Project
from folio.core.schema import InputSpec, OutputSpec
from folio.exceptions import ProjectExistsError, ProjectNotFoundError

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".folio" / "folio.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    inputs_json TEXT NOT NULL,
    outputs_json TEXT NOT NULL,
    target_configs_json TEXT NOT NULL,
    reference_point_json TEXT,
    recommender_config_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    inputs_json TEXT NOT NULL,
    outputs_json TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    notes TEXT,
    tag TEXT,
    raw_data_path TEXT,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_observations_project_id ON observations(project_id);
"""


def _ensure_db_dir(db_path: Path) -> None:
    """Create database directory if it doesn't exist.

    Parameters
    ----------
    db_path : Path
        Path to the database file. Parent directories will be created if needed.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)


def init_db(
    db_path: Path = DEFAULT_DB_PATH,
    sync_url: str | None = None,
    auth_token: str | None = None,
) -> None:
    """Initialize the database with schema tables.

    Creates the projects and observations tables if they don't exist.
    Safe to call multiple times; existing tables are not modified.

    Parameters
    ----------
    db_path : Path, optional
        Path to the database file. Defaults to ~/.folio/folio.db.
    sync_url : str, optional
        libSQL sync URL for cloud synchronization (e.g., libsql://your-db.turso.io).
        If provided, uses libsql.connect() instead of sqlite3.connect().
    auth_token : str, optional
        Authentication token for libSQL cloud sync. Required if sync_url is provided.

    Examples
    --------
    >>> # Local SQLite
    >>> init_db(Path("my_project.db"))

    >>> # libSQL with cloud sync
    >>> init_db(
    ...     Path("my_project.db"),
    ...     sync_url="libsql://my-db.turso.io",
    ...     auth_token="my-token"
    ... )
    """
    _ensure_db_dir(db_path)
    if sync_url is not None:
        conn = libsql.connect(str(db_path), sync_url=sync_url, auth_token=auth_token)
        conn.executescript(_SCHEMA)
        conn.commit()
        conn.sync()
        conn.close()
    else:
        with sqlite3.connect(db_path) as conn:
            conn.executescript(_SCHEMA)
    logger.info(f"Initialized database at {db_path}")


@contextmanager
def get_connection(
    db_path: Path = DEFAULT_DB_PATH,
    sync_url: str | None = None,
    auth_token: str | None = None,
) -> Iterator[Any]:
    """Context manager for database connections with automatic commit/rollback.

    Opens a connection with foreign keys enabled and Row factory configured.
    Automatically commits on successful exit or rolls back on exception.
    For libSQL connections, calls sync() after commit to push changes to cloud.

    Parameters
    ----------
    db_path : Path, optional
        Path to the database file. Defaults to ~/.folio/folio.db.
    sync_url : str, optional
        libSQL sync URL for cloud synchronization (e.g., libsql://your-db.turso.io).
        If provided, uses libsql.connect() instead of sqlite3.connect().
    auth_token : str, optional
        Authentication token for libSQL cloud sync. Required if sync_url is provided.

    Yields
    ------
    sqlite3.Connection | libsql.Connection
        Database connection with Row factory enabled.

    Examples
    --------
    >>> # Local SQLite
    >>> with get_connection() as conn:
    ...     conn.execute("SELECT * FROM projects")

    >>> # libSQL with cloud sync
    >>> with get_connection(
    ...     sync_url="libsql://my-db.turso.io",
    ...     auth_token="my-token"
    ... ) as conn:
    ...     conn.execute("INSERT INTO projects ...")
    """
    _ensure_db_dir(db_path)
    use_libsql = sync_url is not None

    if use_libsql:
        conn = libsql.connect(str(db_path), sync_url=sync_url, auth_token=auth_token)
    else:
        conn = sqlite3.connect(db_path)

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
        if use_libsql:
            conn.sync()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _serialize_inputs(inputs: list[InputSpec]) -> str:
    """Serialize InputSpec list to JSON string for database storage.

    Parameters
    ----------
    inputs : list[InputSpec]
        Input specifications to serialize.

    Returns
    -------
    str
        JSON string representation.
    """
    return json.dumps([asdict(inp) for inp in inputs])


def _deserialize_inputs(inputs_json: str) -> list[InputSpec]:
    """Deserialize JSON string to InputSpec list.

    Parameters
    ----------
    inputs_json : str
        JSON string from database.

    Returns
    -------
    list[InputSpec]
        Reconstructed input specifications.
    """
    return [InputSpec(**d) for d in json.loads(inputs_json)]


def _serialize_outputs(outputs: list[OutputSpec]) -> str:
    """Serialize OutputSpec list to JSON string for database storage.

    Parameters
    ----------
    outputs : list[OutputSpec]
        Output specifications to serialize.

    Returns
    -------
    str
        JSON string representation.
    """
    return json.dumps([asdict(out) for out in outputs])


def _deserialize_outputs(outputs_json: str) -> list[OutputSpec]:
    """Deserialize JSON string to OutputSpec list.

    Parameters
    ----------
    outputs_json : str
        JSON string from database.

    Returns
    -------
    list[OutputSpec]
        Reconstructed output specifications.
    """
    return [OutputSpec(**d) for d in json.loads(outputs_json)]


def _serialize_target_configs(configs: list[TargetConfig]) -> str:
    """Serialize list of TargetConfig to JSON string for database storage.

    Parameters
    ----------
    configs : list[TargetConfig]
        Target configurations to serialize.

    Returns
    -------
    str
        JSON string representation.
    """
    return json.dumps([asdict(config) for config in configs])


def _deserialize_target_configs(configs_json: str) -> list[TargetConfig]:
    """Deserialize JSON string to list of TargetConfig.

    Parameters
    ----------
    configs_json : str
        JSON string from database.

    Returns
    -------
    list[TargetConfig]
        Reconstructed target configurations.
    """
    return [TargetConfig(**d) for d in json.loads(configs_json)]


def _serialize_reference_point(ref_point: list[float] | None) -> str | None:
    """Serialize reference_point to JSON string for database storage.

    Parameters
    ----------
    ref_point : list[float] | None
        Reference point to serialize.

    Returns
    -------
    str | None
        JSON string representation, or None if ref_point is None.
    """
    if ref_point is None:
        return None
    return json.dumps(ref_point)


def _deserialize_reference_point(ref_point_json: str | None) -> list[float] | None:
    """Deserialize JSON string to reference_point.

    Parameters
    ----------
    ref_point_json : str | None
        JSON string from database.

    Returns
    -------
    list[float] | None
        Reconstructed reference point, or None if input is None.
    """
    if ref_point_json is None:
        return None
    return json.loads(ref_point_json)


def _serialize_recommender_config(config: RecommenderConfig) -> str:
    """Serialize RecommenderConfig to JSON string for database storage.

    Parameters
    ----------
    config : RecommenderConfig
        Recommender configuration to serialize.

    Returns
    -------
    str
        JSON string representation.
    """
    return json.dumps(asdict(config))


def _deserialize_recommender_config(config_json: str) -> RecommenderConfig:
    """Deserialize JSON string to RecommenderConfig.

    Parameters
    ----------
    config_json : str
        JSON string from database.

    Returns
    -------
    RecommenderConfig
        Reconstructed recommender configuration.
    """
    return RecommenderConfig(**json.loads(config_json))


def _row_to_project(row: sqlite3.Row) -> Project:
    """Convert database row to Project instance.

    Parameters
    ----------
    row : sqlite3.Row
        Database row from projects table.

    Returns
    -------
    Project
        Reconstructed project with all fields populated.
    """
    return Project(
        id=row["id"],
        name=row["name"],
        inputs=_deserialize_inputs(row["inputs_json"]),
        outputs=_deserialize_outputs(row["outputs_json"]),
        target_configs=_deserialize_target_configs(row["target_configs_json"]),
        reference_point=_deserialize_reference_point(row["reference_point_json"]),
        recommender_config=_deserialize_recommender_config(
            row["recommender_config_json"]
        ),
    )


def create_project(
    project: Project,
    db_path: Path = DEFAULT_DB_PATH,
    sync_url: str | None = None,
    auth_token: str | None = None,
) -> Project:
    """Create a new project in the database.

    Parameters
    ----------
    project : Project
        Project to create. The id field is ignored; a new ID will be assigned.
    db_path : Path, optional
        Path to the database file. Defaults to ~/.folio/folio.db.
    sync_url : str, optional
        libSQL sync URL for cloud synchronization.
    auth_token : str, optional
        Authentication token for libSQL cloud sync.

    Returns
    -------
    Project
        The created project with its assigned database ID.

    Raises
    ------
    ProjectExistsError
        If a project with the same name already exists.
    """
    init_db(db_path, sync_url=sync_url, auth_token=auth_token)
    with get_connection(db_path, sync_url=sync_url, auth_token=auth_token) as conn:
        try:
            cursor = conn.execute(
                """
                INSERT INTO projects (name, inputs_json, outputs_json,
                                      target_configs_json, reference_point_json,
                                      recommender_config_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    project.name,
                    _serialize_inputs(project.inputs),
                    _serialize_outputs(project.outputs),
                    _serialize_target_configs(project.target_configs),
                    _serialize_reference_point(project.reference_point),
                    _serialize_recommender_config(project.recommender_config),
                ),
            )
            project_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            raise ProjectExistsError(
                f"Project '{project.name}' already exists. "
                "Use a different name or delete the existing project."
            )

    logger.info(f"Created project '{project.name}' with id {project_id}")
    return Project(
        id=project_id,
        name=project.name,
        inputs=project.inputs,
        outputs=project.outputs,
        target_configs=project.target_configs,
        reference_point=project.reference_point,
        recommender_config=project.recommender_config,
    )


def get_project(
    name: str,
    db_path: Path = DEFAULT_DB_PATH,
    sync_url: str | None = None,
    auth_token: str | None = None,
) -> Project:
    """Get a project by name.

    Parameters
    ----------
    name : str
        Name of the project to retrieve.
    db_path : Path, optional
        Path to the database file. Defaults to ~/.folio/folio.db.
    sync_url : str, optional
        libSQL sync URL for cloud synchronization.
    auth_token : str, optional
        Authentication token for libSQL cloud sync.

    Returns
    -------
    Project
        The requested project.

    Raises
    ------
    ProjectNotFoundError
        If no project with the given name exists.
    """
    init_db(db_path, sync_url=sync_url, auth_token=auth_token)
    with get_connection(db_path, sync_url=sync_url, auth_token=auth_token) as conn:
        row = conn.execute("SELECT * FROM projects WHERE name = ?", (name,)).fetchone()

    if row is None:
        available = list_projects(db_path, sync_url=sync_url, auth_token=auth_token)
        raise ProjectNotFoundError(f"No project named '{name}'. Available: {available}")

    return _row_to_project(row)


def get_project_by_id(
    project_id: int,
    db_path: Path = DEFAULT_DB_PATH,
    sync_url: str | None = None,
    auth_token: str | None = None,
) -> Project:
    """Get a project by database ID.

    Parameters
    ----------
    project_id : int
        Database ID of the project to retrieve.
    db_path : Path, optional
        Path to the database file. Defaults to ~/.folio/folio.db.
    sync_url : str, optional
        libSQL sync URL for cloud synchronization.
    auth_token : str, optional
        Authentication token for libSQL cloud sync.

    Returns
    -------
    Project
        The requested project.

    Raises
    ------
    ProjectNotFoundError
        If no project with the given ID exists.
    """
    init_db(db_path, sync_url=sync_url, auth_token=auth_token)
    with get_connection(db_path, sync_url=sync_url, auth_token=auth_token) as conn:
        row = conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()

    if row is None:
        raise ProjectNotFoundError(f"No project with id {project_id}")

    return _row_to_project(row)


def list_projects(
    db_path: Path = DEFAULT_DB_PATH,
    sync_url: str | None = None,
    auth_token: str | None = None,
) -> list[str]:
    """List all project names in alphabetical order.

    Parameters
    ----------
    db_path : Path, optional
        Path to the database file. Defaults to ~/.folio/folio.db.
    sync_url : str, optional
        libSQL sync URL for cloud synchronization.
    auth_token : str, optional
        Authentication token for libSQL cloud sync.

    Returns
    -------
    list[str]
        Names of all projects, sorted alphabetically.
    """
    init_db(db_path, sync_url=sync_url, auth_token=auth_token)
    with get_connection(db_path, sync_url=sync_url, auth_token=auth_token) as conn:
        rows = conn.execute("SELECT name FROM projects ORDER BY name").fetchall()
    return [row["name"] for row in rows]


def delete_project(
    name: str,
    db_path: Path = DEFAULT_DB_PATH,
    sync_url: str | None = None,
    auth_token: str | None = None,
) -> None:
    """Delete a project and all its observations.

    Parameters
    ----------
    name : str
        Name of the project to delete.
    db_path : Path, optional
        Path to the database file. Defaults to ~/.folio/folio.db.
    sync_url : str, optional
        libSQL sync URL for cloud synchronization.
    auth_token : str, optional
        Authentication token for libSQL cloud sync.

    Raises
    ------
    ProjectNotFoundError
        If no project with the given name exists.

    Notes
    -----
    All observations associated with the project are deleted via CASCADE.
    """
    with get_connection(db_path, sync_url=sync_url, auth_token=auth_token) as conn:
        cursor = conn.execute("DELETE FROM projects WHERE name = ?", (name,))
        if cursor.rowcount == 0:
            available = list_projects(db_path, sync_url=sync_url, auth_token=auth_token)
            raise ProjectNotFoundError(
                f"No project named '{name}'. Available: {available}"
            )
    logger.info(f"Deleted project '{name}'")


def _row_to_observation(row: sqlite3.Row) -> Observation:
    """Convert database row to Observation instance.

    Parameters
    ----------
    row : sqlite3.Row
        Database row from observations table.

    Returns
    -------
    Observation
        Reconstructed observation with all fields populated.
    """
    timestamp = row["timestamp"]
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)
    return Observation(
        id=row["id"],
        project_id=row["project_id"],
        inputs=json.loads(row["inputs_json"]),
        outputs=json.loads(row["outputs_json"]),
        timestamp=timestamp,
        notes=row["notes"],
        tag=row["tag"],
        raw_data_path=row["raw_data_path"],
    )


def add_observation(
    observation: Observation,
    db_path: Path = DEFAULT_DB_PATH,
    sync_url: str | None = None,
    auth_token: str | None = None,
) -> Observation:
    """Add an observation to the database.

    Parameters
    ----------
    observation : Observation
        Observation to add. The id field is ignored; a new ID will be assigned.
    db_path : Path, optional
        Path to the database file. Defaults to ~/.folio/folio.db.
    sync_url : str, optional
        libSQL sync URL for cloud synchronization.
    auth_token : str, optional
        Authentication token for libSQL cloud sync.

    Returns
    -------
    Observation
        The added observation with its assigned database ID.
    """
    with get_connection(db_path, sync_url=sync_url, auth_token=auth_token) as conn:
        cursor = conn.execute(
            """
            INSERT INTO observations (project_id, inputs_json, outputs_json,
                                      timestamp, notes, tag, raw_data_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observation.project_id,
                json.dumps(observation.inputs),
                json.dumps(observation.outputs),
                observation.timestamp.isoformat(),
                observation.notes,
                observation.tag,
                observation.raw_data_path,
            ),
        )
        observation_id = cursor.lastrowid

    logger.info(
        f"Added observation {observation_id} to project {observation.project_id}"
    )
    return Observation(
        id=observation_id,
        project_id=observation.project_id,
        inputs=observation.inputs,
        outputs=observation.outputs,
        timestamp=observation.timestamp,
        notes=observation.notes,
        tag=observation.tag,
        raw_data_path=observation.raw_data_path,
    )


def get_observations(
    project_id: int,
    db_path: Path = DEFAULT_DB_PATH,
    sync_url: str | None = None,
    auth_token: str | None = None,
) -> list[Observation]:
    """Get all observations for a project, ordered by timestamp.

    Parameters
    ----------
    project_id : int
        Database ID of the project.
    db_path : Path, optional
        Path to the database file. Defaults to ~/.folio/folio.db.
    sync_url : str, optional
        libSQL sync URL for cloud synchronization.
    auth_token : str, optional
        Authentication token for libSQL cloud sync.

    Returns
    -------
    list[Observation]
        All observations for the project, sorted chronologically.
    """
    with get_connection(db_path, sync_url=sync_url, auth_token=auth_token) as conn:
        rows = conn.execute(
            """
            SELECT * FROM observations
            WHERE project_id = ?
            ORDER BY timestamp
            """,
            (project_id,),
        ).fetchall()
    return [_row_to_observation(row) for row in rows]


def get_observation(
    observation_id: int,
    db_path: Path = DEFAULT_DB_PATH,
    sync_url: str | None = None,
    auth_token: str | None = None,
) -> Observation:
    """Get a single observation by database ID.

    Parameters
    ----------
    observation_id : int
        Database ID of the observation to retrieve.
    db_path : Path, optional
        Path to the database file. Defaults to ~/.folio/folio.db.
    sync_url : str, optional
        libSQL sync URL for cloud synchronization.
    auth_token : str, optional
        Authentication token for libSQL cloud sync.

    Returns
    -------
    Observation
        The requested observation.

    Raises
    ------
    ValueError
        If no observation with the given ID exists.
    """
    with get_connection(db_path, sync_url=sync_url, auth_token=auth_token) as conn:
        row = conn.execute(
            "SELECT * FROM observations WHERE id = ?", (observation_id,)
        ).fetchone()

    if row is None:
        raise ValueError(f"No observation with id {observation_id}")

    return _row_to_observation(row)


def delete_observation(
    observation_id: int,
    db_path: Path = DEFAULT_DB_PATH,
    sync_url: str | None = None,
    auth_token: str | None = None,
) -> None:
    """Delete an observation by database ID.

    Parameters
    ----------
    observation_id : int
        Database ID of the observation to delete.
    db_path : Path, optional
        Path to the database file. Defaults to ~/.folio/folio.db.
    sync_url : str, optional
        libSQL sync URL for cloud synchronization.
    auth_token : str, optional
        Authentication token for libSQL cloud sync.

    Raises
    ------
    ValueError
        If no observation with the given ID exists.
    """
    with get_connection(db_path, sync_url=sync_url, auth_token=auth_token) as conn:
        cursor = conn.execute(
            "DELETE FROM observations WHERE id = ?", (observation_id,)
        )
        if cursor.rowcount == 0:
            raise ValueError(f"No observation with id {observation_id}")
    logger.info(f"Deleted observation {observation_id}")


def count_observations(
    project_id: int,
    db_path: Path = DEFAULT_DB_PATH,
    sync_url: str | None = None,
    auth_token: str | None = None,
) -> int:
    """Count the number of observations for a project.

    Parameters
    ----------
    project_id : int
        Database ID of the project.
    db_path : Path, optional
        Path to the database file. Defaults to ~/.folio/folio.db.
    sync_url : str, optional
        libSQL sync URL for cloud synchronization.
    auth_token : str, optional
        Authentication token for libSQL cloud sync.

    Returns
    -------
    int
        Number of observations recorded for the project.
    """
    with get_connection(db_path, sync_url=sync_url, auth_token=auth_token) as conn:
        row = conn.execute(
            "SELECT COUNT(*) as count FROM observations WHERE project_id = ?",
            (project_id,),
        ).fetchone()
    return row["count"]
