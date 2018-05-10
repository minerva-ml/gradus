import unittest
import numpy as np

from steps.base import Step, Dummy, StepsError

from .steps_test_case import CACHE_DIRPATH, remove_cache


class StepTest(unittest.TestCase):
    def setUp(self):
        remove_cache()

        self.data = {
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

    def tearDown(self):
        remove_cache()


    def test_inputs_without_conflicting_names_do_not_require_adapter(self):
        step = Step(
            name='test_inputs_without_conflicting_names_do_not_require_adapter_1',
            transformer=Dummy(),
            input_data=['input_1'],
            cache_dirpath=CACHE_DIRPATH
        )
        output = step.fit_transform(self.data)
        assert output == self.data['input_1']

        step = Step(
            name='test_inputs_without_conflicting_names_do_not_require_adapter_2',
            transformer=Dummy(),
            input_data=['input_1', 'input_2'],
            cache_dirpath=CACHE_DIRPATH
        )
        output = step.fit_transform(self.data)
        assert output == {**self.data['input_1'], **self.data['input_2']}

    def test_inputs_with_conflicting_names_require_adapter(self):
        step = Step(
            name='test_inputs_with_conflicting_names_require_adapter',
            transformer=Dummy(),
            input_data=['input_1', 'input_3'],
            cache_dirpath=CACHE_DIRPATH
        )
        with self.assertRaises(StepsError):
            step.fit_transform(self.data)

    def test_adapter_creates_defined_keys(self):
        step = Step(
            name='test_adapter_creates_defined_keys',
            transformer=Dummy(),
            input_data=['input_1', 'input_2'],
            adapter={'X': [('input_1', 'features')],
                     'Y': [('input_2', 'extra_features')]},
            cache_dirpath=CACHE_DIRPATH
        )
        res = step.fit_transform(self.data)
        self.assertEqual({'X', 'Y'}, set(res.keys()))

    def test_adapter_recipe_with_single_item(self):
        step = Step(
            name='test_adapter_recipe_with_single_item',
            transformer=Dummy(),
            input_data=['input_1'],
            adapter={'X': ('input_1', 'features')},
            cache_dirpath=CACHE_DIRPATH
        )
        res = step.fit_transform(self.data)
        self.assertTrue(np.array_equal(res['X'], self.data['input_1']['features']))

    def test_adapter_recipe_with_list(self):
        step = Step(
            name='test_adapter_recipe_with_list',
            transformer=Dummy(),
            input_data=['input_1', 'input_2'],
            adapter={'X': [],
                     'Y': [('input_1', 'features')],
                     'Z': [('input_1', 'features'), ('input_2', 'extra_features')]},
            cache_dirpath=CACHE_DIRPATH
        )
        res = step.fit_transform(self.data)
        for i, key in enumerate(('X', 'Y', 'Z')):
            self.assertIsInstance(res[key], list)
            self.assertEqual(len(res[key]), i)

        self.assertEqual(res['X'], [])

        self.assertTrue(np.array_equal(res['Y'][0], self.data['input_1']['features']))

        self.assertTrue(np.array_equal(res['Z'][0], self.data['input_1']['features']))
        self.assertTrue(np.array_equal(res['Z'][1], self.data['input_2']['extra_features']))

    def test_adapter_recipe_with_list_and_function(self):
        step = Step(
            name='test_adapter_recipe_with_list_and_function',
            transformer=Dummy(),
            input_data=['input_1', 'input_2'],
            adapter={'X': ([('input_1', 'features'),
                            ('input_2', 'extra_features')],
                           lambda lst: np.hstack(lst))
                     },
            cache_dirpath=CACHE_DIRPATH
        )
        res = step.fit_transform(self.data)
        expected = np.hstack([self.data['input_1']['features'],
                              self.data['input_2']['extra_features']])
        self.assertTrue(np.array_equal(res['X'], expected))
