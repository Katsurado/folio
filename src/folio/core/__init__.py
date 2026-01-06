"""Core data models and database operations."""

from folio.core.database import (
    add_observation,
    count_observations,
    create_project,
    delete_observation,
    delete_project,
    get_observation,
    get_observations,
    get_project,
    get_project_by_id,
    init_db,
    list_projects,
)
from folio.core.observation import Observation
from folio.core.project import Project, RecommenderConfig, TargetConfig
from folio.core.schema import InputSpec, OutputSpec

__all__ = [
    "InputSpec",
    "OutputSpec",
    "Observation",
    "Project",
    "TargetConfig",
    "RecommenderConfig",
    "init_db",
    "create_project",
    "get_project",
    "get_project_by_id",
    "list_projects",
    "delete_project",
    "add_observation",
    "get_observations",
    "get_observation",
    "delete_observation",
    "count_observations",
]
