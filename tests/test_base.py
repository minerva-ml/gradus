import pytest
import numpy as np

from steps.base import Step, Dummy, StepsError

from .steps_test_utils import CACHE_DIRPATH


@pytest.fixture
def data():
    return {
        'input_1': {
            'features': np.array([
                [1, 6],
                [2, 5],
                [3, 4]
            ]),
            'labels': np.array([2, 5, 3])
        },
        'input_2': {
            'extra_features': np.array([
                [5, 7, 3],
                [67, 4, 5],
                [6, 13, 14]
            ])
        },
        'input_3': {
            'images': np.array([
                [[0, 255], [255, 0]],
                [[255, 0], [0, 255]],
                [[255, 255], [0, 0]],
            ]),
            'labels': np.array([1, 1, 0])
        }
    }


def test_inputs_without_conflicting_names_do_not_require_adapter(data):
    step = Step(
        name='test_inputs_without_conflicting_names_do_not_require_adapter_1',
        transformer=Dummy(),
        input_data=['input_1'],
        cache_dirpath=CACHE_DIRPATH
    )
    output = step.fit_transform(data)
    assert output == data['input_1']

    step = Step(
        name='test_inputs_without_conflicting_names_do_not_require_adapter_2',
        transformer=Dummy(),
        input_data=['input_1', 'input_2'],
        cache_dirpath=CACHE_DIRPATH
    )
    output = step.fit_transform(data)
    assert output == {**data['input_1'], **data['input_2']}


def test_inputs_with_conflicting_names_require_adapter(data):
    step = Step(
        name='test_inputs_with_conflicting_names_require_adapter',
        transformer=Dummy(),
        input_data=['input_1', 'input_3'],
        cache_dirpath=CACHE_DIRPATH
    )
    with pytest.raises(StepsError):
        step.fit_transform(data)
