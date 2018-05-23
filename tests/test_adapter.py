import numpy as np
import pytest

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


def test_adapter_creates_defined_keys(data):
    adapter = Adapter({
        'X': [E('input_1', 'features')],
        'Y': [E('input_2', 'extra_features')]
    })
    res = adapter.adapt(data)

    assert {'X', 'Y'} == set(res.keys())


def test_recipe_with_single_item(data):
    adapter = Adapter({
        'X': E('input_1', 'labels'),
        'Y': E('input_3', 'labels'),
    })
    res = adapter.adapt(data)

    assert np.array_equal(res['X'], data['input_1']['labels'])
    assert np.array_equal(res['Y'], data['input_3']['labels'])


def test_recipe_with_list(data):
    adapter = Adapter({
        'X': [],
        'Y': [E('input_1', 'features')],
        'Z': [E('input_1', 'features'),
              E('input_2', 'extra_features')]
    })
    res = adapter.adapt(data)
    for i, key in enumerate(('X', 'Y', 'Z')):
        assert isinstance(res[key], list)
        assert len(res[key]) == i

    assert res['X'] == []
    assert np.array_equal(res['Y'][0], data['input_1']['features'])
    assert np.array_equal(res['Z'][0], data['input_1']['features'])
    assert np.array_equal(res['Z'][1], data['input_2']['extra_features'])


def test_recipe_with_tuple(data):
    adapter = Adapter({
        'X': (),
        'Y': (E('input_1', 'features'),),
        'Z': (E('input_1', 'features'), E('input_2', 'extra_features'))
    })
    res = adapter.adapt(data)

    for i, key in enumerate(('X', 'Y', 'Z')):
        assert isinstance(res[key], tuple)
        assert len(res[key]) == i

    assert res['X'] == ()
    assert np.array_equal(res['Y'][0], data['input_1']['features'])
    assert np.array_equal(res['Z'][0], data['input_1']['features'])
    assert np.array_equal(res['Z'][1], data['input_2']['extra_features'])


def test_recipe_with_dictionary(data):
    adapter = Adapter({
        'X': {},
        'Y': {'a': E('input_1', 'features')},
        'Z': {'a': E('input_1', 'features'),
              'b': E('input_2', 'extra_features')}
    })
    res = adapter.adapt(data)

    for i, key in enumerate(('X', 'Y', 'Z')):
        assert isinstance(res[key], dict)
        assert len(res[key]) == i

    assert res['X'] == {}
    assert np.array_equal(res['Y']['a'], data['input_1']['features'])
    assert np.array_equal(res['Z']['a'], data['input_1']['features'])
    assert np.array_equal(res['Z']['b'], data['input_2']['extra_features'])


def test_recipe_with_constants(data):
    adapter = Adapter({
        'A': 112358,
        'B': 3.14,
        'C': "lorem ipsum",
        'D': ('input_1', 'features'),
        'E': {112358: 112358, 'a': 'a', 3.14: 3.14},
        'F': [112358, 3.14, "lorem ipsum", ('input_1', 'features')]
    })
    res = adapter.adapt(data)

    assert res['A'] == 112358
    assert res['B'] == 3.14
    assert res['C'] == "lorem ipsum"
    assert res['D'] == ('input_1', 'features')
    assert res['E'] == {112358: 112358, 'a': 'a', 3.14: 3.14}
    assert res['F'] == [112358, 3.14, "lorem ipsum", ('input_1', 'features')]


def test_nested_recipes(data):
    adapter = Adapter({
        'X': [{'a': [E('input_1', 'features')]}],
        'Y': {'a': [{'b': E('input_2', 'extra_features')}]}
    })
    res = adapter.adapt(data)

    assert res['X'] == [{'a': [data['input_1']['features']]}]
    assert res['Y'] == {'a': [{'b': data['input_2']['extra_features']}]}
