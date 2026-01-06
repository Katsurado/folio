"""SQLite database operations for projects and observations."""

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from folio.core.observation import Observation
from folio.core.project import Project, RecommenderConfig, TargetConfig
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
    target_config_json TEXT NOT NULL,
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
    """Create database directory if it doesn't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Initialize the database with schema."""
    _ensure_db_dir(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA)
    logger.info(f"Initialized database at {db_path}")


@contextmanager
def get_connection(db_path: Path = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    """Context manager for database connections."""
    _ensure_db_dir(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _serialize_inputs(inputs: list[InputSpec]) -> str:
    """Serialize InputSpec list to JSON."""
    return json.dumps([asdict(inp) for inp in inputs])


def _deserialize_inputs(inputs_json: str) -> list[InputSpec]:
    """Deserialize JSON to InputSpec list."""
    return [InputSpec(**d) for d in json.loads(inputs_json)]


def _serialize_outputs(outputs: list[OutputSpec]) -> str:
    """Serialize OutputSpec list to JSON."""
    return json.dumps([asdict(out) for out in outputs])


def _deserialize_outputs(outputs_json: str) -> list[OutputSpec]:
    """Deserialize JSON to OutputSpec list."""
    return [OutputSpec(**d) for d in json.loads(outputs_json)]


def _serialize_target_config(config: TargetConfig) -> str:
    """Serialize TargetConfig to JSON."""
    return json.dumps(asdict(config))


def _deserialize_target_config(config_json: str) -> TargetConfig:
    """Deserialize JSON to TargetConfig."""
    return TargetConfig(**json.loads(config_json))


def _serialize_recommender_config(config: RecommenderConfig) -> str:
    """Serialize RecommenderConfig to JSON."""
    return json.dumps(asdict(config))


def _deserialize_recommender_config(config_json: str) -> RecommenderConfig:
    """Deserialize JSON to RecommenderConfig."""
    return RecommenderConfig(**json.loads(config_json))


def _row_to_project(row: sqlite3.Row) -> Project:
    """Convert database row to Project."""
    return Project(
        id=row["id"],
        name=row["name"],
        inputs=_deserialize_inputs(row["inputs_json"]),
        outputs=_deserialize_outputs(row["outputs_json"]),
        target_config=_deserialize_target_config(row["target_config_json"]),
        recommender_config=_deserialize_recommender_config(
            row["recommender_config_json"]
        ),
    )


def create_project(project: Project, db_path: Path = DEFAULT_DB_PATH) -> Project:
    """Create a new project in the database."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        try:
            cursor = conn.execute(
                """
                INSERT INTO projects (name, inputs_json, outputs_json,
                                      target_config_json, recommender_config_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    project.name,
                    _serialize_inputs(project.inputs),
                    _serialize_outputs(project.outputs),
                    _serialize_target_config(project.target_config),
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
        target_config=project.target_config,
        recommender_config=project.recommender_config,
    )


def get_project(name: str, db_path: Path = DEFAULT_DB_PATH) -> Project:
    """Get a project by name."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM projects WHERE name = ?", (name,)).fetchone()

    if row is None:
        available = list_projects(db_path)
        raise ProjectNotFoundError(f"No project named '{name}'. Available: {available}")

    return _row_to_project(row)


def get_project_by_id(project_id: int, db_path: Path = DEFAULT_DB_PATH) -> Project:
    """Get a project by ID."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()

    if row is None:
        raise ProjectNotFoundError(f"No project with id {project_id}")

    return _row_to_project(row)


def list_projects(db_path: Path = DEFAULT_DB_PATH) -> list[str]:
    """List all project names."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT name FROM projects ORDER BY name").fetchall()
    return [row["name"] for row in rows]


def delete_project(name: str, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Delete a project and all its observations."""
    with get_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM projects WHERE name = ?", (name,))
        if cursor.rowcount == 0:
            available = list_projects(db_path)
            raise ProjectNotFoundError(
                f"No project named '{name}'. Available: {available}"
            )
    logger.info(f"Deleted project '{name}'")


def _row_to_observation(row: sqlite3.Row) -> Observation:
    """Convert database row to Observation."""
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
    observation: Observation, db_path: Path = DEFAULT_DB_PATH
) -> Observation:
    """Add an observation to the database."""
    with get_connection(db_path) as conn:
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
    project_id: int, db_path: Path = DEFAULT_DB_PATH
) -> list[Observation]:
    """Get all observations for a project, ordered by timestamp."""
    with get_connection(db_path) as conn:
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
    observation_id: int, db_path: Path = DEFAULT_DB_PATH
) -> Observation:
    """Get a single observation by ID."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM observations WHERE id = ?", (observation_id,)
        ).fetchone()

    if row is None:
        raise ValueError(f"No observation with id {observation_id}")

    return _row_to_observation(row)


def delete_observation(observation_id: int, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Delete an observation by ID."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "DELETE FROM observations WHERE id = ?", (observation_id,)
        )
        if cursor.rowcount == 0:
            raise ValueError(f"No observation with id {observation_id}")
    logger.info(f"Deleted observation {observation_id}")


def count_observations(project_id: int, db_path: Path = DEFAULT_DB_PATH) -> int:
    """Count observations for a project."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) as count FROM observations WHERE project_id = ?",
            (project_id,),
        ).fetchone()
    return row["count"]
