import numpy as np
import pytest

from steppy.base import Step, StepsError, make_transformer, IdentityOperation
from steppy.adapter import Adapter, E


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
    def fun(x, y, m=0):
        return x + y if m == 0 else x - y
    tr = make_transformer(fun)

    tr.fit()
    res = tr.transform(7, 3, m=mode)
    assert res == (10 if mode == 0 else 4)


def test_inputs_without_conflicting_names_do_not_require_adapter(data):
    step = Step(
        name='test_inputs_without_conflicting_names_do_not_require_adapter_1',
        transformer=IdentityOperation(),
        input_data=['input_1']
    )
    output = step.fit_transform(data)
    assert output == data['input_1']

    step = Step(
        name='test_inputs_without_conflicting_names_do_not_require_adapter_2',
        transformer=IdentityOperation(),
        input_data=['input_1', 'input_2']
    )
    output = step.fit_transform(data)
    assert output == {**data['input_1'], **data['input_2']}


def test_inputs_with_conflicting_names_require_adapter(data):
    step = Step(
        name='test_inputs_with_conflicting_names_require_adapter',
        transformer=IdentityOperation(),
        input_data=['input_1', 'input_3']
    )
    with pytest.raises(StepsError):
        step.fit_transform(data)


def test_step_with_adapted_inputs(data):
    step = Step(
        name='test_step_wit_adapted_inputs',
        transformer=IdentityOperation(),
        input_data=['input_1', 'input_3'],
        adapter=Adapter({
            'img': E('input_3', 'images'),
            'fea': E('input_1', 'features'),
            'l1': E('input_3', 'labels'),
            'l2': E('input_1', 'labels'),
        })
    )
    output = step.fit_transform(data)
    expected = {
        'img': data['input_3']['images'],
        'fea': data['input_1']['features'],
        'l1': data['input_3']['labels'],
        'l2': data['input_1']['labels'],
    }
    assert output == expected
