import pytest

from powermet.demo import DemoSpec, generate


@pytest.fixture(scope="session")
def demo_df():
    return generate(DemoSpec(n_designs=2, n_builds=6, n_fubs=20, seed=1))
