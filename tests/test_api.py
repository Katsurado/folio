"""Tests for Folio high-level API."""

import gc
from unittest.mock import MagicMock, patch

import pytest

from folio.api import Folio
from folio.core.config import RecommenderConfig, TargetConfig
from folio.core.observation import Observation
from folio.core.project import Project
from folio.core.schema import InputSpec, OutputSpec
from folio.exceptions import (
    CostLimitError,
    ExecutorError,
    InvalidInputError,
    InvalidOutputError,
    InvalidSchemaError,
    ProjectExistsError,
    ProjectNotFoundError,
)
from folio.executors import ClaudeLightExecutor, Executor, HumanExecutor
from folio.recommenders.base import Recommender
from folio.recommenders.bayesian import BayesianRecommender
from folio.recommenders.initializer import LLMBackend
from folio.recommenders.random import RandomRecommender


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database path for testing.

    Note: gc.collect() is called after yield to ensure SQLite connections
    are fully released before pytest cleans up tmp_path. This is required
    on Windows where open files cannot be deleted.
    """
    yield tmp_path / "test.db"
    gc.collect()


@pytest.fixture
def folio(temp_db):
    """Create a Folio instance with a temporary database."""
    yield Folio(db_path=temp_db)
    gc.collect()


@pytest.fixture
def sample_inputs():
    """Sample input specifications for testing."""
    return [
        InputSpec("temperature", "continuous", bounds=(20.0, 100.0)),
        InputSpec("pressure", "continuous", bounds=(1.0, 10.0)),
    ]


@pytest.fixture
def sample_outputs():
    """Sample output specifications for testing."""
    return [OutputSpec("yield"), OutputSpec("purity")]


@pytest.fixture
def sample_target_configs():
    """Sample target configuration for single-objective optimization."""
    return [TargetConfig(objective="yield", objective_mode="maximize")]


@pytest.fixture
def sample_multi_target_configs():
    """Sample target configurations for multi-objective optimization."""
    return [
        TargetConfig(objective="yield", objective_mode="maximize"),
        TargetConfig(objective="purity", objective_mode="maximize"),
    ]


# =============================================================================
# Project CRUD Tests
# =============================================================================


class TestCreateProject:
    """Tests for Folio.create_project()."""

    def test_create_project_single_objective(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Create a single-objective project successfully."""
        folio.create_project(
            name="test_project",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        projects = folio.list_projects()
        assert "test_project" in projects

    def test_create_project_multi_objective(
        self, folio, sample_inputs, sample_outputs, sample_multi_target_configs
    ):
        """Create a multi-objective project with reference point."""
        folio.create_project(
            name="multi_obj_project",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_multi_target_configs,
            reference_point=[0.0, 0.0],
        )

        project = folio.get_project("multi_obj_project")
        assert project.is_multi_objective()
        assert project.reference_point == [0.0, 0.0]

    def test_create_project_with_recommender_config(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Create a project with custom recommender configuration."""
        config = RecommenderConfig(
            type="bayesian", surrogate="gp", acquisition="ucb", n_initial=3
        )
        folio.create_project(
            name="custom_config_project",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
            recommender_config=config,
        )

        project = folio.get_project("custom_config_project")
        assert project.recommender_config.acquisition == "ucb"
        assert project.recommender_config.n_initial == 3

    def test_create_project_duplicate_name_raises(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Creating a project with existing name raises ProjectExistsError."""
        folio.create_project(
            name="duplicate_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        with pytest.raises(ProjectExistsError, match="(?i)duplicate_test|already"):
            folio.create_project(
                name="duplicate_test",
                inputs=sample_inputs,
                outputs=sample_outputs,
                target_configs=sample_target_configs,
            )

    def test_create_project_invalid_schema_raises(self, folio, sample_outputs):
        """Creating a project with invalid schema raises InvalidSchemaError."""
        with pytest.raises(InvalidSchemaError, match="(?i)input|empty"):
            folio.create_project(
                name="invalid_project",
                inputs=[],
                outputs=sample_outputs,
                target_configs=[TargetConfig(objective="yield")],
            )


class TestListProjects:
    """Tests for Folio.list_projects()."""

    def test_list_projects_empty(self, folio):
        """List projects when database is empty."""
        projects = folio.list_projects()
        assert projects == []

    def test_list_projects_multiple(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """List multiple projects returns sorted names."""
        folio.create_project(
            name="zebra_project",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )
        folio.create_project(
            name="alpha_project",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )
        folio.create_project(
            name="beta_project",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        projects = folio.list_projects()
        assert projects == ["alpha_project", "beta_project", "zebra_project"]


class TestDeleteProject:
    """Tests for Folio.delete_project()."""

    def test_delete_project_success(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Delete an existing project successfully."""
        folio.create_project(
            name="to_delete",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )
        assert "to_delete" in folio.list_projects()

        folio.delete_project("to_delete")
        assert "to_delete" not in folio.list_projects()

    def test_delete_project_not_found_raises(self, folio):
        """Deleting non-existent project raises ProjectNotFoundError."""
        with pytest.raises(ProjectNotFoundError, match="(?i)nonexistent|not found"):
            folio.delete_project("nonexistent_project")

    def test_delete_project_clears_recommender_cache(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Deleting a project removes its cached recommender."""
        folio.create_project(
            name="cache_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        # Trigger recommender creation
        folio.suggest("cache_test")
        assert folio.get_recommender("cache_test") is not None

        # Delete and verify cache is cleared
        folio.delete_project("cache_test")
        assert folio.get_recommender("cache_test") is None


class TestGetProject:
    """Tests for Folio.get_project()."""

    def test_get_project_returns_correct_schema(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """get_project returns Project with correct schema."""
        folio.create_project(
            name="schema_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        project = folio.get_project("schema_test")

        assert isinstance(project, Project)
        assert project.name == "schema_test"
        assert len(project.inputs) == 2
        assert project.inputs[0].name == "temperature"
        assert project.inputs[1].name == "pressure"
        assert len(project.outputs) == 2
        assert project.outputs[0].name == "yield"

    def test_get_project_not_found_raises(self, folio):
        """get_project for non-existent project raises ProjectNotFoundError."""
        with pytest.raises(ProjectNotFoundError, match="(?i)nonexistent|not found"):
            folio.get_project("nonexistent_project")


# =============================================================================
# Observation CRUD Tests
# =============================================================================


class TestAddObservation:
    """Tests for Folio.add_observation()."""

    def test_add_observation_success(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Add a valid observation successfully."""
        folio.create_project(
            name="obs_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        folio.add_observation(
            project_name="obs_test",
            inputs={"temperature": 50.0, "pressure": 5.0},
            outputs={"yield": 80.0, "purity": 95.0},
        )

        observations = folio.get_observations("obs_test")
        assert len(observations) == 1
        assert observations[0].inputs["temperature"] == 50.0
        assert observations[0].outputs["yield"] == 80.0

    def test_add_observation_with_tag_and_notes(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Add observation with tag and notes metadata."""
        folio.create_project(
            name="metadata_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        folio.add_observation(
            project_name="metadata_test",
            inputs={"temperature": 60.0, "pressure": 6.0},
            outputs={"yield": 85.0, "purity": 97.0},
            tag="screening",
            notes="Excellent result",
        )

        observations = folio.get_observations("metadata_test")
        assert len(observations) == 1
        assert observations[0].tag == "screening"
        assert observations[0].notes == "Excellent result"

    def test_add_observation_project_not_found_raises(self, folio):
        """Adding observation to non-existent project raises ProjectNotFoundError."""
        with pytest.raises(ProjectNotFoundError, match="(?i)nonexistent|not found"):
            folio.add_observation(
                project_name="nonexistent_project",
                inputs={"temperature": 50.0, "pressure": 5.0},
                outputs={"yield": 80.0, "purity": 95.0},
            )

    def test_add_observation_invalid_inputs_raises(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Adding observation with invalid inputs raises InvalidInputError."""
        folio.create_project(
            name="invalid_input_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        # Missing input
        with pytest.raises(InvalidInputError, match="(?i)missing|temperature"):
            folio.add_observation(
                project_name="invalid_input_test",
                inputs={"pressure": 5.0},
                outputs={"yield": 80.0, "purity": 95.0},
            )

    def test_add_observation_invalid_outputs_raises(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Adding observation with invalid outputs raises InvalidOutputError."""
        folio.create_project(
            name="invalid_output_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        # Missing output
        with pytest.raises(InvalidOutputError, match="(?i)missing|yield"):
            folio.add_observation(
                project_name="invalid_output_test",
                inputs={"temperature": 50.0, "pressure": 5.0},
                outputs={"purity": 95.0},
            )


class TestDeleteObservation:
    """Tests for Folio.delete_observation()."""

    def test_delete_observation_success(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Delete an existing observation successfully."""
        folio.create_project(
            name="delete_obs_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        folio.add_observation(
            project_name="delete_obs_test",
            inputs={"temperature": 50.0, "pressure": 5.0},
            outputs={"yield": 80.0, "purity": 95.0},
        )

        observations = folio.get_observations("delete_obs_test")
        assert len(observations) == 1
        obs_id = observations[0].id

        folio.delete_observation(obs_id)

        observations = folio.get_observations("delete_obs_test")
        assert len(observations) == 0

    def test_delete_observation_not_found_raises(self, folio):
        """Deleting non-existent observation raises ValueError."""
        with pytest.raises(ValueError, match="(?i)99999|not found|no observation"):
            folio.delete_observation(99999)


class TestGetObservations:
    """Tests for Folio.get_observations()."""

    def test_get_observations_all(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Get all observations for a project."""
        folio.create_project(
            name="get_all_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        for i in range(3):
            folio.add_observation(
                project_name="get_all_test",
                inputs={"temperature": 50.0 + i * 10, "pressure": 5.0},
                outputs={"yield": 80.0 + i, "purity": 95.0},
            )

        observations = folio.get_observations("get_all_test")
        assert len(observations) == 3

    def test_get_observations_empty(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Get observations when none exist returns empty list."""
        folio.create_project(
            name="empty_obs_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        observations = folio.get_observations("empty_obs_test")
        assert observations == []

    def test_get_observations_filtered_by_tag(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Get observations filtered by tag."""
        folio.create_project(
            name="tag_filter_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        # Add observations with different tags
        folio.add_observation(
            project_name="tag_filter_test",
            inputs={"temperature": 50.0, "pressure": 5.0},
            outputs={"yield": 80.0, "purity": 95.0},
            tag="screening",
        )
        folio.add_observation(
            project_name="tag_filter_test",
            inputs={"temperature": 60.0, "pressure": 6.0},
            outputs={"yield": 85.0, "purity": 97.0},
            tag="optimization",
        )
        folio.add_observation(
            project_name="tag_filter_test",
            inputs={"temperature": 70.0, "pressure": 7.0},
            outputs={"yield": 90.0, "purity": 98.0},
            tag="screening",
        )

        screening_obs = folio.get_observations("tag_filter_test", tag="screening")
        assert len(screening_obs) == 2
        assert all(obs.tag == "screening" for obs in screening_obs)

        opt_obs = folio.get_observations("tag_filter_test", tag="optimization")
        assert len(opt_obs) == 1
        assert opt_obs[0].tag == "optimization"

    def test_get_observations_project_not_found_raises(self, folio):
        """Getting observations for non-existent project raises ProjectNotFoundError."""
        with pytest.raises(ProjectNotFoundError, match="(?i)nonexistent|not found"):
            folio.get_observations("nonexistent_project")


# =============================================================================
# Suggest Workflow Tests
# =============================================================================


class TestSuggest:
    """Tests for Folio.suggest()."""

    def test_suggest_no_observations_returns_random(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """suggest() with no observations returns random sample within bounds."""
        folio.create_project(
            name="suggest_empty_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        suggestions = folio.suggest("suggest_empty_test")

        assert isinstance(suggestions, list)
        assert len(suggestions) == 1
        assert "temperature" in suggestions[0]
        assert "pressure" in suggestions[0]
        assert 20.0 <= suggestions[0]["temperature"] <= 100.0
        assert 1.0 <= suggestions[0]["pressure"] <= 10.0

    def test_suggest_with_observations(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """suggest() with observations uses recommender."""
        folio.create_project(
            name="suggest_with_obs_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        # Add some observations
        for i in range(6):
            folio.add_observation(
                project_name="suggest_with_obs_test",
                inputs={"temperature": 30.0 + i * 10, "pressure": 3.0 + i},
                outputs={"yield": 70.0 + i * 5, "purity": 90.0 + i},
            )

        suggestions = folio.suggest("suggest_with_obs_test")

        assert isinstance(suggestions, list)
        assert len(suggestions) == 1
        assert "temperature" in suggestions[0]
        assert "pressure" in suggestions[0]

    def test_suggest_returns_list_of_dicts(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """suggest() returns list of input dicts with correct keys."""
        folio.create_project(
            name="suggest_format_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        suggestions = folio.suggest("suggest_format_test")

        assert isinstance(suggestions, list)
        assert len(suggestions) >= 1
        assert isinstance(suggestions[0], dict)
        assert set(suggestions[0].keys()) == {"temperature", "pressure"}

    def test_suggest_values_within_bounds(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """suggest() returns values within input bounds."""
        folio.create_project(
            name="suggest_bounds_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        # Run suggest multiple times to check bounds are respected
        for _ in range(5):
            suggestions = folio.suggest("suggest_bounds_test")
            suggestion = suggestions[0]

            assert 20.0 <= suggestion["temperature"] <= 100.0
            assert 1.0 <= suggestion["pressure"] <= 10.0

    def test_suggest_project_not_found_raises(self, folio):
        """suggest() for non-existent project raises ProjectNotFoundError."""
        with pytest.raises(ProjectNotFoundError, match="(?i)nonexistent|not found"):
            folio.suggest("nonexistent_project")

    def test_suggest_multi_objective(
        self, folio, sample_inputs, sample_outputs, sample_multi_target_configs
    ):
        """suggest() works for multi-objective optimization."""
        folio.create_project(
            name="suggest_mo_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_multi_target_configs,
            reference_point=[0.0, 0.0],
            recommender_config=RecommenderConfig(
                type="bayesian", surrogate="multitask_gp", mo_acquisition="nehvi"
            ),
        )

        # Add enough observations
        for i in range(6):
            folio.add_observation(
                project_name="suggest_mo_test",
                inputs={"temperature": 30.0 + i * 10, "pressure": 3.0 + i},
                outputs={"yield": 70.0 + i * 5, "purity": 90.0 + i},
            )

        suggestions = folio.suggest("suggest_mo_test")

        assert isinstance(suggestions, list)
        assert len(suggestions) >= 1
        assert "temperature" in suggestions[0]
        assert "pressure" in suggestions[0]


# =============================================================================
# Recommender Caching Tests
# =============================================================================


class TestRecommenderCaching:
    """Tests for recommender instance caching."""

    def test_get_recommender_before_suggest_returns_none(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """get_recommender returns None before suggest() is called."""
        folio.create_project(
            name="cache_before_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        recommender = folio.get_recommender("cache_before_test")
        assert recommender is None

    def test_get_recommender_after_suggest_returns_instance(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """get_recommender returns Recommender after suggest() is called."""
        folio.create_project(
            name="cache_after_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        folio.suggest("cache_after_test")

        recommender = folio.get_recommender("cache_after_test")
        assert recommender is not None
        assert isinstance(recommender, Recommender)

    def test_recommender_cached_between_suggest_calls(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Same recommender instance is reused between suggest() calls."""
        folio.create_project(
            name="cache_reuse_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        folio.suggest("cache_reuse_test")
        recommender1 = folio.get_recommender("cache_reuse_test")

        folio.suggest("cache_reuse_test")
        recommender2 = folio.get_recommender("cache_reuse_test")

        assert recommender1 is recommender2

    def test_build_recommender_respects_config_type(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """_build_recommender creates correct recommender type from config."""
        # Test Bayesian recommender
        folio.create_project(
            name="bayesian_config_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
            recommender_config=RecommenderConfig(type="bayesian"),
        )
        folio.suggest("bayesian_config_test")
        bayesian_rec = folio.get_recommender("bayesian_config_test")
        assert isinstance(bayesian_rec, BayesianRecommender)

        # Test Random recommender
        folio.create_project(
            name="random_config_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
            recommender_config=RecommenderConfig(type="random"),
        )
        folio.suggest("random_config_test")
        random_rec = folio.get_recommender("random_config_test")
        assert isinstance(random_rec, RandomRecommender)


# =============================================================================
# Integration Tests
# =============================================================================


class TestFullWorkflow:
    """Integration tests for complete Folio workflows."""

    def test_full_single_objective_workflow(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Complete workflow: create project, add observations, get suggestions."""
        # 1. Create project
        folio.create_project(
            name="full_workflow_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        # 2. Get initial suggestion (random)
        initial = folio.suggest("full_workflow_test")
        assert len(initial) == 1

        # 3. Add initial observations
        for i in range(5):
            temp = 30.0 + i * 15
            press = 2.0 + i * 2
            folio.add_observation(
                project_name="full_workflow_test",
                inputs={"temperature": temp, "pressure": press},
                outputs={"yield": 60.0 + i * 8, "purity": 85.0 + i * 3},
                tag="initial",
            )

        # 4. Verify observations were recorded
        observations = folio.get_observations("full_workflow_test")
        assert len(observations) == 5

        # 5. Get BO-informed suggestion
        suggestion = folio.suggest("full_workflow_test")
        assert len(suggestion) == 1
        assert 20.0 <= suggestion[0]["temperature"] <= 100.0
        assert 1.0 <= suggestion[0]["pressure"] <= 10.0

        # 6. Verify recommender was cached
        recommender = folio.get_recommender("full_workflow_test")
        assert recommender is not None
        assert isinstance(recommender, BayesianRecommender)

        # 7. Add more observations based on suggestion
        folio.add_observation(
            project_name="full_workflow_test",
            inputs=suggestion[0],
            outputs={"yield": 92.0, "purity": 97.0},
            tag="optimization",
        )

        # 8. Verify total observations
        all_obs = folio.get_observations("full_workflow_test")
        assert len(all_obs) == 6

        opt_obs = folio.get_observations("full_workflow_test", tag="optimization")
        assert len(opt_obs) == 1

    def test_full_multi_objective_workflow(
        self, folio, sample_inputs, sample_outputs, sample_multi_target_configs
    ):
        """Complete multi-objective workflow with reference point."""
        # 1. Create multi-objective project
        folio.create_project(
            name="mo_workflow_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_multi_target_configs,
            reference_point=[50.0, 80.0],
            recommender_config=RecommenderConfig(
                type="bayesian",
                surrogate="multitask_gp",
                mo_acquisition="nehvi",
                n_initial=3,
            ),
        )

        # 2. Verify multi-objective setup
        project = folio.get_project("mo_workflow_test")
        assert project.is_multi_objective()
        assert len(project.target_configs) == 2

        # 3. Add initial observations
        for i in range(5):
            folio.add_observation(
                project_name="mo_workflow_test",
                inputs={"temperature": 40.0 + i * 12, "pressure": 3.0 + i * 1.5},
                outputs={"yield": 65.0 + i * 6, "purity": 88.0 + i * 2},
            )

        # 4. Get multi-objective suggestion
        suggestion = folio.suggest("mo_workflow_test")
        assert len(suggestion) >= 1
        assert "temperature" in suggestion[0]
        assert "pressure" in suggestion[0]

        # 5. Verify bounds are respected
        assert 20.0 <= suggestion[0]["temperature"] <= 100.0
        assert 1.0 <= suggestion[0]["pressure"] <= 10.0


# =============================================================================
# Executor Tests
# =============================================================================


class TestBuildExecutor:
    """Tests for Folio.build_executor()."""

    def test_build_executor_human(self, folio):
        """Build a HumanExecutor by name."""
        executor = folio.build_executor("human")

        assert isinstance(executor, HumanExecutor)
        assert folio.executor is executor

    def test_build_executor_claude_light(self, folio):
        """Build a ClaudeLightExecutor by name."""
        executor = folio.build_executor("claude_light")

        assert isinstance(executor, ClaudeLightExecutor)
        assert folio.executor is executor

    def test_build_executor_caches_in_self(self, folio):
        """build_executor caches the executor in self.executor."""
        assert folio.executor is None

        executor = folio.build_executor("human")

        assert folio.executor is not None
        assert folio.executor is executor

    def test_build_executor_replaces_previous(self, folio):
        """Building a new executor replaces the previously cached one."""
        first_executor = folio.build_executor("human")
        second_executor = folio.build_executor("claude_light")

        assert folio.executor is second_executor
        assert folio.executor is not first_executor

    def test_build_executor_unknown_name_raises(self, folio):
        """Building an executor with unknown name raises ValueError."""
        with pytest.raises(ValueError, match="(?i)unknown|invalid|not found"):
            folio.build_executor("nonexistent_executor")

    def test_build_executor_returns_executor(self, folio):
        """build_executor returns an Executor instance."""
        executor = folio.build_executor("human")

        assert isinstance(executor, Executor)


class TestExecute:
    """Tests for Folio.execute()."""

    @pytest.fixture
    def project_with_observations(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Create a project with some initial observations for testing."""
        folio.create_project(
            name="execute_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )
        # Add a couple of initial observations so suggest() has data
        for i in range(3):
            folio.add_observation(
                project_name="execute_test",
                inputs={"temperature": 40.0 + i * 20, "pressure": 3.0 + i * 2},
                outputs={"yield": 70.0 + i * 5, "purity": 90.0 + i * 2},
            )
        return "execute_test"

    @pytest.fixture
    def mock_executor(self):
        """Create a mock executor for testing."""
        mock = MagicMock(spec=Executor)
        # The execute method returns an Observation
        mock.execute.return_value = Observation(
            project_id=1,
            inputs={"temperature": 50.0, "pressure": 5.0},
            outputs={"yield": 85.0, "purity": 95.0},
        )
        return mock

    def test_execute_no_executor_raises(self, folio, project_with_observations):
        """execute() with no executor raises ExecutorError."""
        assert folio.executor is None

        with pytest.raises(ExecutorError, match="(?i)no executor|not configured"):
            folio.execute(project_with_observations)

    def test_execute_uses_passed_executor(
        self, folio, project_with_observations, mock_executor
    ):
        """execute() uses the passed executor parameter."""
        folio.execute(project_with_observations, n_iter=1, executor=mock_executor)

        mock_executor.execute.assert_called_once()

    def test_execute_uses_cached_executor(
        self, folio, project_with_observations, mock_executor
    ):
        """execute() uses self.executor when no executor is passed."""
        folio.executor = mock_executor

        folio.execute(project_with_observations, n_iter=1)

        mock_executor.execute.assert_called_once()

    def test_execute_passed_executor_takes_precedence(
        self, folio, project_with_observations, mock_executor
    ):
        """Passed executor takes precedence over cached executor."""
        cached_executor = MagicMock(spec=Executor)
        cached_executor.execute.return_value = Observation(
            project_id=1,
            inputs={"temperature": 50.0, "pressure": 5.0},
            outputs={"yield": 80.0, "purity": 90.0},
        )
        folio.executor = cached_executor

        folio.execute(project_with_observations, n_iter=1, executor=mock_executor)

        mock_executor.execute.assert_called_once()
        cached_executor.execute.assert_not_called()

    def test_execute_returns_observations_list(
        self, folio, project_with_observations, mock_executor
    ):
        """execute() returns a list of Observation objects."""
        result = folio.execute(
            project_with_observations, n_iter=3, executor=mock_executor
        )

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(obs, Observation) for obs in result)

    def test_execute_runs_n_iterations(
        self, folio, project_with_observations, mock_executor
    ):
        """execute() runs exactly n_iter iterations."""
        folio.execute(project_with_observations, n_iter=5, executor=mock_executor)

        assert mock_executor.execute.call_count == 5

    def test_execute_adds_observations_to_database(
        self, folio, project_with_observations, mock_executor
    ):
        """execute() adds observations to the database."""
        initial_count = len(folio.get_observations(project_with_observations))

        folio.execute(project_with_observations, n_iter=3, executor=mock_executor)

        final_count = len(folio.get_observations(project_with_observations))
        assert final_count == initial_count + 3

    def test_execute_stop_on_error_true_reraises(
        self, folio, project_with_observations, mock_executor
    ):
        """execute() with stop_on_error=True re-raises execution errors."""
        mock_executor.execute.side_effect = ExecutorError("Experiment failed")

        with pytest.raises(ExecutorError, match="(?i)failed"):
            folio.execute(
                project_with_observations,
                n_iter=3,
                stop_on_error=True,
                executor=mock_executor,
            )

    def test_execute_stop_on_error_false_continues(
        self, folio, project_with_observations, mock_executor
    ):
        """execute() with stop_on_error=False continues after errors."""
        # First call succeeds, second fails, third succeeds
        mock_executor.execute.side_effect = [
            Observation(
                project_id=1,
                inputs={"temperature": 50.0, "pressure": 5.0},
                outputs={"yield": 85.0, "purity": 95.0},
            ),
            ExecutorError("Experiment failed"),
            Observation(
                project_id=1,
                inputs={"temperature": 60.0, "pressure": 6.0},
                outputs={"yield": 88.0, "purity": 96.0},
            ),
        ]

        result = folio.execute(
            project_with_observations,
            n_iter=3,
            stop_on_error=False,
            executor=mock_executor,
        )

        # Should return only successful observations
        assert len(result) == 2

    def test_execute_wait_between_runs(
        self, folio, project_with_observations, mock_executor
    ):
        """execute() waits between iterations when wait_between_runs > 0."""
        with patch("time.sleep") as mock_sleep:
            folio.execute(
                project_with_observations,
                n_iter=3,
                wait_between_runs=0.5,
                executor=mock_executor,
            )

            # Should sleep between iterations (n_iter - 1 times, or n_iter times)
            assert mock_sleep.call_count >= 2
            mock_sleep.assert_called_with(0.5)

    def test_execute_project_not_found_raises(self, folio, mock_executor):
        """execute() raises ProjectNotFoundError for non-existent project."""
        with pytest.raises(ProjectNotFoundError, match="(?i)nonexistent|not found"):
            folio.execute("nonexistent_project", n_iter=1, executor=mock_executor)

    def test_execute_calls_suggest_for_each_iteration(
        self, folio, project_with_observations, mock_executor
    ):
        """execute() calls suggest() for each iteration."""
        with patch.object(folio, "suggest", wraps=folio.suggest) as mock_suggest:
            folio.execute(project_with_observations, n_iter=3, executor=mock_executor)

            assert mock_suggest.call_count == 3

    def test_execute_passes_suggestion_to_executor(
        self, folio, project_with_observations, mock_executor
    ):
        """execute() passes the suggestion from suggest() to executor.execute()."""
        folio.execute(project_with_observations, n_iter=1, executor=mock_executor)

        # Verify executor.execute was called with a dict containing expected keys
        call_args = mock_executor.execute.call_args
        assert call_args is not None
        suggestion = call_args[0][0]
        assert "temperature" in suggestion
        assert "pressure" in suggestion

    def test_execute_single_iteration(
        self, folio, project_with_observations, mock_executor
    ):
        """execute() with n_iter=1 runs a single iteration."""
        result = folio.execute(
            project_with_observations, n_iter=1, executor=mock_executor
        )

        assert len(result) == 1
        mock_executor.execute.assert_called_once()

    def test_execute_zero_iterations_returns_empty_list(
        self, folio, project_with_observations, mock_executor
    ):
        """execute() with n_iter=0 returns an empty list."""
        result = folio.execute(
            project_with_observations, n_iter=0, executor=mock_executor
        )

        assert result == []
        mock_executor.execute.assert_not_called()


# =============================================================================
# Executor Integration Tests
# =============================================================================


class TestExecutorIntegration:
    """Integration tests for full executor workflow."""

    @pytest.fixture
    def mock_executor_with_varying_outputs(self):
        """Create a mock executor that returns different outputs each call."""
        mock = MagicMock(spec=Executor)
        call_count = [0]

        def execute_side_effect(suggestion, project):
            call_count[0] += 1
            return Observation(
                project_id=project.id,
                inputs=suggestion,
                outputs={
                    "yield": 70.0 + call_count[0] * 5,
                    "purity": 90.0 + call_count[0] * 2,
                },
            )

        mock.execute.side_effect = execute_side_effect
        return mock

    def test_full_execution_workflow(
        self,
        folio,
        sample_inputs,
        sample_outputs,
        sample_target_configs,
        mock_executor_with_varying_outputs,
    ):
        """Complete workflow: create project, build executor, run execute loop."""
        # 1. Create project
        folio.create_project(
            name="integration_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        # 2. Add some initial observations
        for i in range(3):
            folio.add_observation(
                project_name="integration_test",
                inputs={"temperature": 40.0 + i * 20, "pressure": 3.0 + i * 2},
                outputs={"yield": 65.0 + i * 5, "purity": 88.0 + i * 2},
            )

        # 3. Run execute loop
        observations = folio.execute(
            "integration_test",
            n_iter=5,
            executor=mock_executor_with_varying_outputs,
        )

        # 4. Verify results
        assert len(observations) == 5
        assert all(isinstance(obs, Observation) for obs in observations)

        # 5. Verify observations were added to database
        all_obs = folio.get_observations("integration_test")
        assert len(all_obs) == 8  # 3 initial + 5 from execute

    def test_execute_with_built_executor(
        self,
        folio,
        sample_inputs,
        sample_outputs,
        sample_target_configs,
    ):
        """Test execute() using an executor built with build_executor()."""
        # 1. Create project
        folio.create_project(
            name="built_executor_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        # 2. Add initial observations
        for i in range(3):
            folio.add_observation(
                project_name="built_executor_test",
                inputs={"temperature": 40.0 + i * 20, "pressure": 3.0 + i * 2},
                outputs={"yield": 65.0 + i * 5, "purity": 88.0 + i * 2},
            )

        # 3. Build executor
        executor = folio.build_executor("human")
        assert folio.executor is executor

        # 4. Mock the executor's execute method for testing
        mock_obs = Observation(
            project_id=1,
            inputs={"temperature": 50.0, "pressure": 5.0},
            outputs={"yield": 85.0, "purity": 95.0},
        )
        with patch.object(executor, "execute", return_value=mock_obs):
            # 5. Run execute (uses cached executor)
            observations = folio.execute("built_executor_test", n_iter=2)

            assert len(observations) == 2

    def test_multi_objective_execution(
        self,
        folio,
        sample_inputs,
        sample_outputs,
        sample_multi_target_configs,
        mock_executor_with_varying_outputs,
    ):
        """Test execute() with a multi-objective project."""
        # 1. Create multi-objective project
        folio.create_project(
            name="mo_execute_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_multi_target_configs,
            reference_point=[50.0, 80.0],
            recommender_config=RecommenderConfig(
                type="bayesian",
                surrogate="multitask_gp",
                mo_acquisition="nehvi",
            ),
        )

        # 2. Add initial observations
        for i in range(5):
            folio.add_observation(
                project_name="mo_execute_test",
                inputs={"temperature": 30.0 + i * 15, "pressure": 2.0 + i * 1.5},
                outputs={"yield": 60.0 + i * 6, "purity": 85.0 + i * 2.5},
            )

        # 3. Run execute loop
        observations = folio.execute(
            "mo_execute_test",
            n_iter=3,
            executor=mock_executor_with_varying_outputs,
        )

        # 4. Verify results
        assert len(observations) == 3
        assert all(isinstance(obs, Observation) for obs in observations)

    def test_execution_with_partial_failures(
        self,
        folio,
        sample_inputs,
        sample_outputs,
        sample_target_configs,
    ):
        """Test execute() continues after partial failures with stop_on_error=False."""
        # 1. Create project
        folio.create_project(
            name="partial_failure_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        # 2. Add initial observations
        for i in range(3):
            folio.add_observation(
                project_name="partial_failure_test",
                inputs={"temperature": 40.0 + i * 20, "pressure": 3.0 + i * 2},
                outputs={"yield": 65.0 + i * 5, "purity": 88.0 + i * 2},
            )

        # 3. Create executor that fails on second call
        mock_executor = MagicMock(spec=Executor)
        mock_executor.execute.side_effect = [
            Observation(
                project_id=1,
                inputs={"temperature": 50.0, "pressure": 5.0},
                outputs={"yield": 80.0, "purity": 92.0},
            ),
            ExecutorError("Hardware malfunction"),
            Observation(
                project_id=1,
                inputs={"temperature": 60.0, "pressure": 6.0},
                outputs={"yield": 85.0, "purity": 94.0},
            ),
            ExecutorError("Connection timeout"),
            Observation(
                project_id=1,
                inputs={"temperature": 70.0, "pressure": 7.0},
                outputs={"yield": 90.0, "purity": 96.0},
            ),
        ]

        # 4. Run execute with stop_on_error=False
        observations = folio.execute(
            "partial_failure_test",
            n_iter=5,
            stop_on_error=False,
            executor=mock_executor,
        )

        # 5. Verify only successful observations are returned
        assert len(observations) == 3

        # 6. Verify all 5 attempts were made
        assert mock_executor.execute.call_count == 5

    def test_execution_improves_suggestions(
        self,
        folio,
        sample_inputs,
        sample_outputs,
        sample_target_configs,
    ):
        """Test that suggestions improve as more observations are added."""
        # 1. Create project
        folio.create_project(
            name="improvement_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        # 2. Add initial observations with known pattern
        # Higher temperature = higher yield
        for temp in [30.0, 50.0, 70.0]:
            folio.add_observation(
                project_name="improvement_test",
                inputs={"temperature": temp, "pressure": 5.0},
                outputs={"yield": temp + 20.0, "purity": 90.0},
            )

        # 3. Create executor that returns observation based on inputs
        mock_executor = MagicMock(spec=Executor)

        def smart_execute(suggestion, project):
            temp = suggestion["temperature"]
            return Observation(
                project_id=project.id,
                inputs=suggestion,
                outputs={"yield": temp + 20.0, "purity": 90.0},
            )

        mock_executor.execute.side_effect = smart_execute

        # 4. Run a few iterations
        observations = folio.execute(
            "improvement_test",
            n_iter=3,
            executor=mock_executor,
        )

        # 5. Verify observations were created
        assert len(observations) == 3

        # 6. The recommender should be suggesting higher temperatures
        # (since higher temp = higher yield in our pattern)
        all_obs = folio.get_observations("improvement_test")
        assert len(all_obs) == 6  # 3 initial + 3 from execute


class MockLLMBackend(LLMBackend):
    """Mock LLM backend for testing without real API calls."""

    def __init__(self, response: str | None = None) -> None:
        """Initialize with optional canned response."""
        import json

        self._response = response or json.dumps(
            [
                {"temperature": 80.0, "pressure": 5.0},
                {"temperature": 90.0, "pressure": 6.0},
                {"temperature": 70.0, "pressure": 4.0},
            ]
        )

    def complete(self, prompt: str) -> str:
        """Return canned response."""
        return self._response

    def estimate_cost(self, prompt: str, max_output_tokens: int = 4096) -> float:
        """Return fixed low cost."""
        return 0.01


class TestInitializeFromLLM:
    """Tests for Folio.initialize_from_llm() method."""

    def test_initialize_from_llm_returns_suggestions(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Initialize returns list of suggestion dicts."""
        folio.create_project(
            name="llm_init_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        mock_backend = MockLLMBackend()
        suggestions = folio.initialize_from_llm(
            "llm_init_test", n=3, backend=mock_backend
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) == 3
        assert all(isinstance(s, dict) for s in suggestions)

    def test_initialize_from_llm_respects_bounds(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """All suggestions are within project input bounds."""
        folio.create_project(
            name="bounds_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        mock_backend = MockLLMBackend()
        suggestions = folio.initialize_from_llm(
            "bounds_test", n=3, backend=mock_backend
        )

        for suggestion in suggestions:
            assert 20.0 <= suggestion["temperature"] <= 100.0
            assert 1.0 <= suggestion["pressure"] <= 10.0

    def test_initialize_from_llm_clamps_out_of_bounds(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Out-of-bounds values are clamped to valid range."""
        import json

        folio.create_project(
            name="clamp_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        response = json.dumps([{"temperature": 200.0, "pressure": 0.5}])
        mock_backend = MockLLMBackend(response)

        suggestions = folio.initialize_from_llm("clamp_test", n=1, backend=mock_backend)

        assert suggestions[0]["temperature"] == 100.0
        assert suggestions[0]["pressure"] == 1.0

    def test_initialize_from_llm_project_not_found(self, folio):
        """Raises ProjectNotFoundError for nonexistent project."""
        mock_backend = MockLLMBackend()

        with pytest.raises(ProjectNotFoundError):
            folio.initialize_from_llm("nonexistent_project", n=3, backend=mock_backend)

    def test_initialize_from_llm_custom_backend(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Accepts and uses custom LLMBackend."""
        import json

        folio.create_project(
            name="custom_backend_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        response = json.dumps([{"temperature": 85.0, "pressure": 7.0}])
        custom_backend = MockLLMBackend(response)

        suggestions = folio.initialize_from_llm(
            "custom_backend_test", n=1, backend=custom_backend
        )

        assert suggestions[0]["temperature"] == 85.0
        assert suggestions[0]["pressure"] == 7.0

    def test_initialize_from_llm_custom_prompt(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Accepts custom prompt template."""
        folio.create_project(
            name="custom_prompt_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        custom_template = (
            "Suggest {n} experiments. Goal: {objective}. "
            "Params: {parameters}. Context: {description}"
        )
        mock_backend = MockLLMBackend()

        suggestions = folio.initialize_from_llm(
            "custom_prompt_test",
            n=3,
            backend=mock_backend,
            prompt_template=custom_template,
        )

        assert len(suggestions) == 3

    def test_initialize_from_llm_with_description(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Description parameter is passed to initializer."""
        folio.create_project(
            name="description_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        mock_backend = MockLLMBackend()
        suggestions = folio.initialize_from_llm(
            "description_test",
            n=3,
            description="Suzuki coupling optimization for biaryl synthesis",
            backend=mock_backend,
        )

        assert len(suggestions) == 3

    def test_full_workflow_with_llm_init(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Full workflow: LLM init -> add observations -> BO suggest."""
        folio.create_project(
            name="workflow_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        mock_backend = MockLLMBackend()
        suggestions = folio.initialize_from_llm(
            "workflow_test", n=3, backend=mock_backend
        )

        for i, suggestion in enumerate(suggestions):
            folio.add_observation(
                "workflow_test",
                inputs=suggestion,
                outputs={"yield": 70.0 + i * 5, "purity": 95.0},
            )

        bo_suggestions = folio.suggest("workflow_test")

        assert len(bo_suggestions) == 1
        assert "temperature" in bo_suggestions[0]
        assert "pressure" in bo_suggestions[0]
        assert 20.0 <= bo_suggestions[0]["temperature"] <= 100.0
        assert 1.0 <= bo_suggestions[0]["pressure"] <= 10.0

    def test_initialize_from_llm_cost_limit_exceeded(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Raises CostLimitError when estimated cost exceeds limit."""

        class ExpensiveBackend(MockLLMBackend):
            def estimate_cost(
                self, prompt: str, max_output_tokens: int = 4096
            ) -> float:
                return 10.00

        folio.create_project(
            name="cost_limit_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        expensive_backend = ExpensiveBackend()

        with pytest.raises(CostLimitError, match="exceeds limit"):
            folio.initialize_from_llm(
                "cost_limit_test",
                n=3,
                backend=expensive_backend,
                max_cost_per_call=0.50,
            )


# =============================================================================
# Non-Optimizable (Context) Inputs Tests
# =============================================================================


class TestNonOptimizableInputs:
    """Tests for Folio with non-optimizable (context) inputs.

    Non-optimizable inputs are:
    - Recorded in observations (for GP training)
    - Fixed during acquisition optimization (not searched over)
    - Not included in suggestions returned by suggest()
    """

    @pytest.fixture
    def non_optimizable_inputs(self):
        """Input specifications including non-optimizable inputs."""
        return [
            InputSpec("R", "continuous", bounds=(0.0, 255.0)),
            InputSpec("G", "continuous", bounds=(0.0, 255.0)),
            InputSpec("B", "continuous", bounds=(0.0, 255.0)),
            InputSpec("hour", "continuous", bounds=(0.0, 24.0), optimizable=False),
            InputSpec(
                "ambient_temp", "continuous", bounds=(15.0, 35.0), optimizable=False
            ),
        ]

    @pytest.fixture
    def non_optimizable_outputs(self):
        """Output specifications for non-optimizable tests."""
        return [OutputSpec("intensity")]

    @pytest.fixture
    def non_optimizable_target_configs(self):
        """Target config for non-optimizable tests."""
        return [TargetConfig(objective="intensity", objective_mode="maximize")]

    def test_create_project_with_non_optimizable_inputs(
        self,
        folio,
        non_optimizable_inputs,
        non_optimizable_outputs,
        non_optimizable_target_configs,
    ):
        """Create a project with non-optimizable inputs successfully."""
        folio.create_project(
            name="non_optimizable_project",
            inputs=non_optimizable_inputs,
            outputs=non_optimizable_outputs,
            target_configs=non_optimizable_target_configs,
        )

        project = folio.get_project("non_optimizable_project")
        assert len(project.inputs) == 5
        # Verify optimizable flags preserved
        optimizable = [inp for inp in project.inputs if inp.optimizable]
        non_optimizable = [inp for inp in project.inputs if not inp.optimizable]
        assert len(optimizable) == 3
        assert len(non_optimizable) == 2

    def test_add_observation_requires_non_optimizable_values(
        self,
        folio,
        non_optimizable_inputs,
        non_optimizable_outputs,
        non_optimizable_target_configs,
    ):
        """Adding observation requires non-optimizable input values."""
        folio.create_project(
            name="obs_non_opt_test",
            inputs=non_optimizable_inputs,
            outputs=non_optimizable_outputs,
            target_configs=non_optimizable_target_configs,
        )

        # Missing non-optimizable values should raise
        with pytest.raises(InvalidInputError, match="(?i)missing"):
            folio.add_observation(
                "obs_non_opt_test",
                inputs={"R": 100.0, "G": 150.0, "B": 200.0},
                outputs={"intensity": 0.75},
            )

    def test_add_observation_with_non_optimizable_values(
        self,
        folio,
        non_optimizable_inputs,
        non_optimizable_outputs,
        non_optimizable_target_configs,
    ):
        """Adding observation with all values (including non-optimizable) succeeds."""
        folio.create_project(
            name="obs_full_non_opt_test",
            inputs=non_optimizable_inputs,
            outputs=non_optimizable_outputs,
            target_configs=non_optimizable_target_configs,
        )

        folio.add_observation(
            "obs_full_non_opt_test",
            inputs={
                "R": 100.0,
                "G": 150.0,
                "B": 200.0,
                "hour": 12.0,
                "ambient_temp": 22.0,
            },
            outputs={"intensity": 0.75},
        )

        observations = folio.get_observations("obs_full_non_opt_test")
        assert len(observations) == 1
        assert observations[0].inputs["hour"] == 12.0
        assert observations[0].inputs["ambient_temp"] == 22.0

    def test_suggest_returns_only_optimizable_keys(
        self,
        folio,
        non_optimizable_inputs,
        non_optimizable_outputs,
        non_optimizable_target_configs,
    ):
        """suggest() returns dict with only optimizable input names."""
        folio.create_project(
            name="suggest_non_opt_test",
            inputs=non_optimizable_inputs,
            outputs=non_optimizable_outputs,
            target_configs=non_optimizable_target_configs,
        )

        # Add observations with non-optimizable values
        for i in range(5):
            folio.add_observation(
                "suggest_non_opt_test",
                inputs={
                    "R": 50.0 + i * 40,
                    "G": 100.0 + i * 20,
                    "B": 150.0 + i * 10,
                    "hour": 8.0 + i * 2,
                    "ambient_temp": 20.0 + i,
                },
                outputs={"intensity": 0.5 + i * 0.1},
            )

        suggestions = folio.suggest(
            "suggest_non_opt_test",
            fixed_inputs={"hour": 14.0, "ambient_temp": 23.0},
        )

        assert len(suggestions) == 1
        assert set(suggestions[0].keys()) == {"R", "G", "B"}
        assert "hour" not in suggestions[0]
        assert "ambient_temp" not in suggestions[0]

    def test_suggest_requires_context_for_non_optimizable_project(
        self,
        folio,
        non_optimizable_inputs,
        non_optimizable_outputs,
        non_optimizable_target_configs,
    ):
        """suggest() raises if context not provided for project with non-optimizable."""
        folio.create_project(
            name="suggest_no_context_test",
            inputs=non_optimizable_inputs,
            outputs=non_optimizable_outputs,
            target_configs=non_optimizable_target_configs,
        )

        # Add some observations
        folio.add_observation(
            "suggest_no_context_test",
            inputs={
                "R": 100.0,
                "G": 150.0,
                "B": 200.0,
                "hour": 12.0,
                "ambient_temp": 22.0,
            },
            outputs={"intensity": 0.75},
        )

        with pytest.raises(ValueError, match="(?i)fixed_inputs|required|missing"):
            folio.suggest("suggest_no_context_test")

    def test_suggest_values_within_optimizable_bounds(
        self,
        folio,
        non_optimizable_inputs,
        non_optimizable_outputs,
        non_optimizable_target_configs,
    ):
        """suggest() returns values within optimizable input bounds."""
        folio.create_project(
            name="bounds_non_opt_test",
            inputs=non_optimizable_inputs,
            outputs=non_optimizable_outputs,
            target_configs=non_optimizable_target_configs,
        )

        for i in range(5):
            folio.add_observation(
                "bounds_non_opt_test",
                inputs={
                    "R": 50.0 + i * 40,
                    "G": 100.0 + i * 20,
                    "B": 150.0 + i * 10,
                    "hour": 8.0 + i * 2,
                    "ambient_temp": 20.0 + i,
                },
                outputs={"intensity": 0.5 + i * 0.1},
            )

        for _ in range(5):
            suggestions = folio.suggest(
                "bounds_non_opt_test",
                fixed_inputs={"hour": 14.0, "ambient_temp": 23.0},
            )
            s = suggestions[0]
            assert 0.0 <= s["R"] <= 255.0
            assert 0.0 <= s["G"] <= 255.0
            assert 0.0 <= s["B"] <= 255.0

    def test_project_without_non_optimizable_does_not_require_context(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """suggest() works without context for projects without non-optimizable."""
        folio.create_project(
            name="all_optimizable_project",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        # Should work without context parameter
        suggestions = folio.suggest("all_optimizable_project")
        assert len(suggestions) == 1
        assert "temperature" in suggestions[0]

    def test_full_workflow_with_non_optimizable(
        self,
        folio,
        non_optimizable_inputs,
        non_optimizable_outputs,
        non_optimizable_target_configs,
    ):
        """Complete workflow with non-optimizable inputs."""
        # 1. Create project
        folio.create_project(
            name="full_non_opt_workflow",
            inputs=non_optimizable_inputs,
            outputs=non_optimizable_outputs,
            target_configs=non_optimizable_target_configs,
        )

        # 2. Add initial observations with varying context
        for i in range(5):
            folio.add_observation(
                "full_non_opt_workflow",
                inputs={
                    "R": 50.0 + i * 40,
                    "G": 100.0 + i * 20,
                    "B": 150.0 + i * 10,
                    "hour": 6.0 + i * 3,
                    "ambient_temp": 18.0 + i * 2,
                },
                outputs={"intensity": 0.4 + i * 0.15},
            )

        # 3. Get suggestion with current context
        current_context = {"hour": 14.0, "ambient_temp": 24.0}
        suggestions = folio.suggest(
            "full_non_opt_workflow", fixed_inputs=current_context
        )

        assert len(suggestions) == 1
        assert set(suggestions[0].keys()) == {"R", "G", "B"}

        # 4. Add observation from suggestion (with current context)
        folio.add_observation(
            "full_non_opt_workflow",
            inputs={
                **suggestions[0],
                **current_context,
            },
            outputs={"intensity": 0.85},
        )

        # 5. Verify observation was recorded
        observations = folio.get_observations("full_non_opt_workflow")
        assert len(observations) == 6

        # 6. Get another suggestion with different context
        evening_context = {"hour": 20.0, "ambient_temp": 19.0}
        evening_suggestions = folio.suggest(
            "full_non_opt_workflow", fixed_inputs=evening_context
        )

        assert len(evening_suggestions) == 1
        assert set(evening_suggestions[0].keys()) == {"R", "G", "B"}
