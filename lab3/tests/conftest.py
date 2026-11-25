import pytest

from lab3.prepare_data import prepare_official


@pytest.fixture(scope='session', autouse=True)
def div2k_data(tmp_path_factory):
    return prepare_official(tmp_path_factory.mktemp('div2k'), ('0801',))
