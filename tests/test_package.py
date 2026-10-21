import pytest
from pydantic import ValidationError

from buildingmind import __version__
from buildingmind.models import Model


def test_package_version():
    assert __version__ == "0.4.0"


def test_boundary_models_reject_unknown_fields():
    with pytest.raises(ValidationError):
        Model(unexpected="value")
