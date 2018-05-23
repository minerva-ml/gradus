import steppy.base  # To make sure logger is initialized before running prepare_steps_logger

from .steps_test_utils import prepare_steps_logger, remove_cache


def pytest_sessionstart(session):
    prepare_steps_logger()


def pytest_runtest_setup(item):
    remove_cache()


def pytest_runtest_teardown(item):
    remove_cache()
