import steppy.base  # To make sure logger is initialized before running prepare_steps_logger

from .steppy_test_utils import prepare_steps_logger


def pytest_sessionstart(session):
    prepare_steps_logger()


def pytest_runtest_setup(item):
    pass


def pytest_runtest_teardown(item):
    pass
