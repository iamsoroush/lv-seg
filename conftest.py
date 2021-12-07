import pytest

import sys
sys.path.append('src')


def pytest_configure(config):
    """Add custom markers."""

    config.addinivalue_line('markers', 'faketest: faketests for tests tests!')
    config.addinivalue_line('markers', 'component: add this to test your components')
    config.addinivalue_line('markers', 'init: add this to tests that need to run before component tests.')


def pytest_addoption(parser):
    """Get ``run_name`` as argument and use it in a fixture via ``pytestconfig.getoption``."""

    parser.addoption("--run_name", required=False, default='template')
    parser.addoption("--config_path", required=False)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Add each test's docstring as test-id."""

    outcome = yield
    report = outcome.get_result()

    test_fn = item.obj
    docstring = getattr(test_fn, '__doc__')
    if docstring:
        report.nodeid += f' --- {docstring}'
