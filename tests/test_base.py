import numpy as np
import pytest

from steppy.base import Step, IdentityOperation, StepsError, make_transformer
from .steppy_test_utils import EXP_DIR


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


@pytest.mark.parametrize("mode", [0, 1])
def test_make_transformer(mode):
    def fun(x, y, mode=0):
        return x + y if mode == 0 else x - y
    tr = make_transformer(fun)

    tr.fit()
    res = tr.transform(7, 3, mode=mode)
    assert res == (10 if mode == 0 else 4)


def test_inputs_without_conflicting_names_do_not_require_adapter(data):
    step = Step(
        name='test_inputs_without_conflicting_names_do_not_require_adapter_1',
        transformer=IdentityOperation(),
        input_data=['input_1'],
        experiment_directory=EXP_DIR
    )
    output = step.fit_transform(data)
    assert output == data['input_1']

    step = Step(
        name='test_inputs_without_conflicting_names_do_not_require_adapter_2',
        transformer=IdentityOperation(),
        input_data=['input_1', 'input_2'],
        experiment_directory=EXP_DIR
    )
    output = step.fit_transform(data)
    assert output == {**data['input_1'], **data['input_2']}


def test_inputs_with_conflicting_names_require_adapter(data):
    step = Step(
        name='test_inputs_with_conflicting_names_require_adapter',
        transformer=IdentityOperation(),
        input_data=['input_1', 'input_3'],
        experiment_directory=EXP_DIR
    )
    with pytest.raises(StepsError):
        step.fit_transform(data)
