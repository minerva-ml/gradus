import pytest
from millet.core.base import BaseStep
from millet import SupervTransformer, UnsupervTransformer, BaseDataLoader


# Classes for testing


class DummyStep(BaseStep):
    pass


class DummySupervTransformer(SupervTransformer):
    def fit(self, input_dict: dict, superv_dict: dict) -> None:
        pass

    def transform(self, input_dict: dict) -> dict:
        return {}


class DummyUnsupervTransformer(UnsupervTransformer):
    def fit(self, input_dict: dict) -> None:
        pass

    def transform(self, input_dict: dict) -> dict:
        return {}


class DummyDataLoader(BaseDataLoader):
    def load_data(self, input_info: dict) -> dict:
        return {}


# Tests


class TestBaseStep(object):
    def test_subclassing(self):
        step = DummyStep()
        assert isinstance(step, BaseStep)


class TestSupervTransformer(object):
    def test_fit(self):
        step = DummySupervTransformer()
        assert step.fit({}, {}) is None

    def test_transform(self):
        dt = DummySupervTransformer()
        assert dt.transform({}) == {}

    def test_fit_transform(self):
        dt = DummySupervTransformer()
        res = dt.fit_transform({}, {})
        assert res == {}


class TestUnsupervTransformer(object):
    def test_fit(self):
        step = DummyUnsupervTransformer()
        assert step.fit({}) is None

    def test_transform(self):
        dt = DummyUnsupervTransformer()
        assert dt.transform({}) == {}

    def test_fit_transform(self):
        dt = DummyUnsupervTransformer()
        res = dt.fit_transform({})
        assert res == {}


class TestDataLoader(object):
    def test_load_data(self):
        dl = DummyDataLoader()
        assert dl.load_data({}) == {}
