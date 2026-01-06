import pytest

from folio.core.observation import Observation
from folio.core.project import Project, TargetConfig
from folio.core.schema import InputSpec, OutputSpec
from folio.targets import (
    DerivedTarget,
    DifferenceTarget,
    DirectTarget,
    DistanceTarget,
    RatioTarget,
    SlopeTarget,
)


@pytest.fixture
def sample_obs():
    return Observation(
        project_id=1,
        inputs={"temperature": 50.0},
        outputs={"yield": 85.0, "purity": 95.0, "cost": 10.0},
    )


class TestDirectTarget:
    def test_valid_obs(self, sample_obs):
        target = DirectTarget("yield", "maximize")
        assert target.compute(sample_obs) == 85.0
        assert target.objective == "maximize"

    def test_missing_key(self, sample_obs):
        target = DirectTarget("nonexistent", "maximize")
        assert target.compute(sample_obs) is None


class TestDerivedTarget:
    def test_with_lambda(self, sample_obs):
        target = DerivedTarget(
            lambda outputs: outputs["yield"] * outputs["purity"] / 100,
            "maximize",
        )
        assert target.compute(sample_obs) == pytest.approx(80.75)

    def test_exception_returns_none(self, sample_obs):
        def bad_func(outputs):
            raise ValueError("Something went wrong")

        target = DerivedTarget(bad_func, "maximize")
        assert target.compute(sample_obs) is None


class TestRatioTarget:
    def test_valid_obs(self, sample_obs):
        target = RatioTarget("yield", "cost", "maximize")
        assert target.compute(sample_obs) == pytest.approx(8.5)

    def test_missing_key(self, sample_obs):
        target = RatioTarget("yield", "nonexistent", "maximize")
        assert target.compute(sample_obs) is None

    def test_zero_denominator(self):
        obs = Observation(
            project_id=1,
            inputs={"temperature": 50.0},
            outputs={"yield": 85.0, "cost": 0.0},
        )
        target = RatioTarget("yield", "cost", "maximize")
        assert target.compute(obs) is None


class TestDifferenceTarget:
    def test_valid_obs(self, sample_obs):
        target = DifferenceTarget("purity", "yield", "maximize")
        assert target.compute(sample_obs) == pytest.approx(10.0)

    def test_missing_key(self, sample_obs):
        target = DifferenceTarget("purity", "nonexistent", "maximize")
        assert target.compute(sample_obs) is None


class TestSlopeTarget:
    def test_valid_obs(self):
        obs = Observation(
            project_id=1,
            inputs={"temperature": 50.0},
            outputs={"y1": 10.0, "y2": 20.0, "y3": 30.0},
        )
        target = SlopeTarget(["y1", "y2", "y3"], [1.0, 2.0, 3.0], "maximize")
        assert target.compute(obs) == pytest.approx(10.0)

    def test_missing_output(self):
        obs = Observation(
            project_id=1,
            inputs={"temperature": 50.0},
            outputs={"y1": 10.0, "y2": 20.0},
        )
        target = SlopeTarget(["y1", "y2", "y3"], [1.0, 2.0, 3.0], "maximize")
        assert target.compute(obs) is None

    def test_insufficient_points_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            SlopeTarget(["y1", "y2"], [1.0, 2.0], "maximize")

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="must match"):
            SlopeTarget(["y1", "y2", "y3"], [1.0, 2.0], "maximize")


class TestDistanceTarget:
    def test_euclidean_metric(self):
        obs = Observation(
            project_id=1,
            inputs={"temperature": 50.0},
            outputs={"x": 3.0, "y": 4.0},
        )
        target = DistanceTarget(["x", "y"], [0.0, 0.0], "euclidean")
        assert target.compute(obs) == pytest.approx(5.0)
        assert target.objective == "minimize"

    def test_missing_output(self):
        obs = Observation(
            project_id=1,
            inputs={"temperature": 50.0},
            outputs={"x": 3.0},
        )
        target = DistanceTarget(["x", "y"], [0.0, 0.0], "euclidean")
        assert target.compute(obs) is None

    def test_set_target(self):
        obs = Observation(
            project_id=1,
            inputs={"temperature": 50.0},
            outputs={"x": 3.0, "y": 4.0},
        )
        target = DistanceTarget(["x", "y"])
        assert target.compute(obs) is None
        target.set_target([3.0, 4.0])
        assert target.compute(obs) == pytest.approx(0.0)

    def test_mse_metric(self):
        obs = Observation(
            project_id=1,
            inputs={"temperature": 50.0},
            outputs={"x": 1.0, "y": 2.0},
        )
        target = DistanceTarget(["x", "y"], [0.0, 0.0], "mse")
        # MSE = (1^2 + 2^2) / 2 = 2.5
        assert target.compute(obs) == pytest.approx(2.5)

    def test_mae_metric(self):
        obs = Observation(
            project_id=1,
            inputs={"temperature": 50.0},
            outputs={"x": 1.0, "y": 2.0},
        )
        target = DistanceTarget(["x", "y"], [0.0, 0.0], "mae")
        # MAE = (1 + 2) / 2 = 1.5
        assert target.compute(obs) == pytest.approx(1.5)


@pytest.fixture
def sample_project():
    return Project(
        id=None,
        name="test_project",
        inputs=[
            InputSpec(name="temperature", type="continuous", bounds=(20.0, 100.0)),
        ],
        outputs=[OutputSpec(name="yield"), OutputSpec(name="purity")],
        target_config=TargetConfig(name="yield", mode="maximize"),
    )


class TestGetTarget:
    def test_direct_target(self, sample_project):
        target = sample_project.get_target()
        assert isinstance(target, DirectTarget)
        assert target.output_name == "yield"
        assert target.objective == "maximize"

    def test_ratio_target(self):
        project = Project(
            id=None,
            name="test_project",
            inputs=[
                InputSpec(name="temperature", type="continuous", bounds=(20.0, 100.0)),
            ],
            outputs=[OutputSpec(name="yield"), OutputSpec(name="cost")],
            target_config=TargetConfig(
                name="efficiency",
                mode="maximize",
                target_type="ratio",
                numerator="yield",
                denominator="cost",
            ),
        )
        target = project.get_target()
        assert isinstance(target, RatioTarget)
        assert target.numerator == "yield"
        assert target.denominator == "cost"

    def test_difference_target(self):
        project = Project(
            id=None,
            name="test_project",
            inputs=[
                InputSpec(name="temperature", type="continuous", bounds=(20.0, 100.0)),
            ],
            outputs=[OutputSpec(name="yield"), OutputSpec(name="cost")],
            target_config=TargetConfig(
                name="profit",
                mode="maximize",
                target_type="difference",
                first="yield",
                second="cost",
            ),
        )
        target = project.get_target()
        assert isinstance(target, DifferenceTarget)
        assert target.first == "yield"
        assert target.second == "cost"

    def test_slope_target(self):
        project = Project(
            id=None,
            name="test_project",
            inputs=[
                InputSpec(name="temperature", type="continuous", bounds=(20.0, 100.0)),
            ],
            outputs=[
                OutputSpec(name="y1"),
                OutputSpec(name="y2"),
                OutputSpec(name="y3"),
            ],
            target_config=TargetConfig(
                name="slope",
                mode="maximize",
                target_type="slope",
                slope_outputs=["y1", "y2", "y3"],
                slope_x=[1.0, 2.0, 3.0],
            ),
        )
        target = project.get_target()
        assert isinstance(target, SlopeTarget)
        assert target.output_names == ["y1", "y2", "y3"]
        assert target.x_values == [1.0, 2.0, 3.0]


class TestGetTrainingData:
    def test_filters_failed_observations(self, sample_project):
        obs1 = Observation(
            project_id=1,
            inputs={"temperature": 50.0},
            outputs={"yield": 80.0, "purity": 90.0},
        )
        obs2 = Observation(
            project_id=1,
            inputs={"temperature": 60.0},
            outputs={"yield": 85.0, "purity": 92.0},
            failed=True,
        )
        obs3 = Observation(
            project_id=1,
            inputs={"temperature": 70.0},
            outputs={"yield": 90.0, "purity": 95.0},
        )
        X, y = sample_project.get_training_data([obs1, obs2, obs3])
        assert X.shape == (2, 1)
        assert y.shape == (2,)
        assert list(X[:, 0]) == [50.0, 70.0]
        assert list(y) == [80.0, 90.0]

    def test_filters_none_target_values(self):
        project = Project(
            id=None,
            name="test_project",
            inputs=[
                InputSpec(name="temperature", type="continuous", bounds=(20.0, 100.0)),
            ],
            outputs=[OutputSpec(name="yield"), OutputSpec(name="cost")],
            target_config=TargetConfig(
                name="efficiency",
                mode="maximize",
                target_type="ratio",
                numerator="yield",
                denominator="cost",
            ),
        )
        obs1 = Observation(
            project_id=1,
            inputs={"temperature": 50.0},
            outputs={"yield": 80.0, "cost": 10.0},
        )
        obs2 = Observation(
            project_id=1,
            inputs={"temperature": 60.0},
            outputs={"yield": 85.0, "cost": 0.0},
        )
        obs3 = Observation(
            project_id=1,
            inputs={"temperature": 70.0},
            outputs={"yield": 90.0, "cost": 5.0},
        )
        X, y = project.get_training_data([obs1, obs2, obs3])
        assert X.shape == (2, 1)
        assert y.shape == (2,)
        assert list(X[:, 0]) == [50.0, 70.0]
        assert list(y) == pytest.approx([8.0, 18.0])

    def test_empty_observations(self, sample_project):
        X, y = sample_project.get_training_data([])
        assert X.shape == (0,)
        assert y.shape == (0,)
