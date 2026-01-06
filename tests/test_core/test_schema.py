import pytest

from folio.core.schema import ConstantSpec, InputSpec, OutputSpec
from folio.exceptions import InvalidSchemaError


class TestInputSpec:
    def test_continuous_input(self):
        spec = InputSpec(
            name="temperature",
            type="continuous",
            bounds=(20.0, 100.0),
            units="C",
        )
        assert spec.name == "temperature"
        assert spec.type == "continuous"
        assert spec.bounds == (20.0, 100.0)
        assert spec.levels is None
        assert spec.units == "C"

    def test_categorical_input(self):
        spec = InputSpec(
            name="solvent",
            type="categorical",
            levels=["water", "ethanol", "acetone"],
        )
        assert spec.name == "solvent"
        assert spec.type == "categorical"
        assert spec.bounds is None
        assert spec.levels == ["water", "ethanol", "acetone"]
        assert spec.units is None

    def test_continuous_input_equal_bounds_raises(self):
        with pytest.raises(InvalidSchemaError, match="invalid bounds"):
            InputSpec(name="temperature", type="continuous", bounds=(50.0, 50.0))

    def test_continuous_input_inverted_bounds_raises(self):
        with pytest.raises(InvalidSchemaError, match="invalid bounds"):
            InputSpec(name="temperature", type="continuous", bounds=(100.0, 20.0))

    def test_categorical_input_single_level_raises(self):
        with pytest.raises(InvalidSchemaError, match="at least 2 levels"):
            InputSpec(name="solvent", type="categorical", levels=["water"])

    def test_categorical_input_empty_levels_raises(self):
        with pytest.raises(InvalidSchemaError, match="at least 2 levels"):
            InputSpec(name="solvent", type="categorical", levels=[])


class TestOutputSpec:
    def test_output_with_units(self):
        spec = OutputSpec(name="yield", units="%")
        assert spec.name == "yield"
        assert spec.units == "%"

    def test_output_without_units(self):
        spec = OutputSpec(name="selectivity")
        assert spec.name == "selectivity"
        assert spec.units is None


class TestConstantSpec:
    def test_numeric_value_with_units(self):
        spec = ConstantSpec(name="pressure", value=1.0, units="atm")
        assert spec.name == "pressure"
        assert spec.value == 1.0
        assert spec.units == "atm"

    def test_numeric_value_without_units(self):
        spec = ConstantSpec(name="ph", value=7.0)
        assert spec.name == "ph"
        assert spec.value == 7.0
        assert spec.units is None

    def test_string_value_with_units(self):
        spec = ConstantSpec(name="catalyst", value="Pd/C", units="mol%")
        assert spec.name == "catalyst"
        assert spec.value == "Pd/C"
        assert spec.units == "mol%"

    def test_string_value_without_units(self):
        spec = ConstantSpec(name="catalyst", value="Pd/C")
        assert spec.name == "catalyst"
        assert spec.value == "Pd/C"
        assert spec.units is None
