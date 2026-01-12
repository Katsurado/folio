"""Tests for Folio high-level API."""

import pytest

from folio.api import Folio
from folio.core.config import RecommenderConfig, TargetConfig
from folio.core.observation import Observation
from folio.core.project import Project
from folio.core.schema import InputSpec, OutputSpec
from folio.exceptions import (
    InvalidInputError,
    InvalidOutputError,
    InvalidSchemaError,
    ProjectExistsError,
    ProjectNotFoundError,
)
from folio.recommenders.base import Recommender
from folio.recommenders.bayesian import BayesianRecommender
from folio.recommenders.random import RandomRecommender


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database path for testing."""
    return tmp_path / "test.db"


@pytest.fixture
def folio(temp_db):
    """Create a Folio instance with a temporary database."""
    return Folio(db_path=temp_db)


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


class TestLoadProject:
    """Tests for Folio.load_project()."""

    def test_load_project_success(
        self, folio, sample_inputs, sample_outputs, sample_target_configs
    ):
        """Load an existing project successfully."""
        folio.create_project(
            name="load_test",
            inputs=sample_inputs,
            outputs=sample_outputs,
            target_configs=sample_target_configs,
        )

        # load_project should not raise
        folio.load_project("load_test")

        # After loading, recommender should be available
        recommender = folio.get_recommender("load_test")
        assert recommender is not None
        assert isinstance(recommender, Recommender)

    def test_load_project_not_found_raises(self, folio):
        """Loading non-existent project raises ProjectNotFoundError."""
        with pytest.raises(ProjectNotFoundError, match="(?i)nonexistent|not found"):
            folio.load_project("nonexistent_project")


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
