import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from pathlib import Path

from steps.sklearn.models import SklearnClassifier, SklearnRegressor, SklearnTransformer

@pytest.fixture()
def X():
    return np.array([
            [4.2, 3.7, 8.9],
            [3.1, 3.2, 0.5],
            [1.4, 0.9, 8.9],
            [5.8, 5.0, 2.4],
            [5.6, 7.8, 2.4],
            [0.1, 7.0, 0.2],
            [8.3, 1.9, 7.8],
            [3.8, 9.2, 2.8],
            [5.3, 5.7, 4.5],
            [6.8, 5.3, 3.2]
        ])


@pytest.fixture()
def y():
    return np.array([4, 0, 0, 1, 0, 4, 2, 3, 2, 1])


@pytest.mark.parametrize(
    "sklearn_class,steps_wrapper,transform_method", [
        (RandomForestClassifier, SklearnClassifier, 'predict_proba'),
        (RandomForestRegressor, SklearnRegressor, 'predict'),
        (PCA, SklearnTransformer, 'transform'),
    ]
)
def test_fit_transform(X, y, sklearn_class, steps_wrapper, transform_method):
    tr = steps_wrapper(sklearn_class(random_state=11235813))
    tr.fit(X, y)
    tr_pred = tr.transform(X)[steps_wrapper.RESULT_KEY]
    rf = sklearn_class(random_state=11235813)
    rf.fit(X, y)
    rf_pred = getattr(rf, transform_method)(X)
    assert np.array_equal(tr_pred, rf_pred)


@pytest.mark.parametrize(
    "sklearn_class,steps_wrapper", [
        (RandomForestClassifier, SklearnClassifier),
        (RandomForestRegressor, SklearnRegressor),
        (PCA, SklearnTransformer),
    ]
)
def test_save_load(X, y, tmpdir, sklearn_class, steps_wrapper):
    tr = steps_wrapper(sklearn_class())
    tr.fit(X, y)
    before = tr.transform(X)[steps_wrapper.RESULT_KEY]
    print("Tmp dir:'{}'".format(tmpdir))
    path = str(Path(str(tmpdir)) / 'transformer.tmp')
    tr.save(path)
    loaded_tr = steps_wrapper(sklearn_class())
    loaded_tr.load(path)
    after = loaded_tr.transform(X)[steps_wrapper.RESULT_KEY]
    assert np.array_equal(before, after)
