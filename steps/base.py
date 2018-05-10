import os
import pprint
import shutil
from collections import defaultdict

from sklearn.externals import joblib

from .utils import view_graph, plot_graph, get_logger, initialize_logger

initialize_logger()
logger = get_logger()


class Step:
    def __init__(self, name, transformer, input_steps=[], input_data=[], adapter=None,
                 cache_dirpath=None, cache_output=False, save_output=False, load_saved_output=False,
                 save_graph=False, force_fitting=False):
        self.name = name

        self.transformer = transformer

        self.input_steps = input_steps
        self.input_data = input_data
        self.adapter = adapter

        self.force_fitting = force_fitting
        self.cache_output = cache_output
        self.save_output = save_output
        self.load_saved_output = load_saved_output

        self.cache_dirpath = cache_dirpath
        self._prep_cache(cache_dirpath)

        if save_graph:
            graph_filepath = os.path.join(self.cache_dirpath, '{}_graph.json'.format(self.name))
            logger.info('Saving graph to {}'.format(graph_filepath))
            joblib.dump(self.graph_info, graph_filepath)

    def _copy_transformer(self, step, name, dirpath):
        self.transformer = self.transformer.transformer

        original_filepath = os.path.join(step.cache_dirpath, 'transformers', step.name)
        copy_filepath = os.path.join(dirpath, 'transformers', name)
        logger.info('copying transformer from {} to {}'.format(original_filepath, copy_filepath))
        shutil.copyfile(original_filepath, copy_filepath)

    def _prep_cache(self, cache_dirpath):
        for dirname in ['transformers', 'outputs', 'tmp']:
            os.makedirs(os.path.join(cache_dirpath, dirname), exist_ok=True)

        self.cache_dirpath_transformers = os.path.join(cache_dirpath, 'transformers')
        self.save_dirpath_outputs = os.path.join(cache_dirpath, 'outputs')
        self.save_dirpath_tmp = os.path.join(cache_dirpath, 'tmp')

        self.cache_filepath_step_transformer = os.path.join(self.cache_dirpath_transformers, self.name)
        self.save_filepath_step_output = os.path.join(self.save_dirpath_outputs, '{}'.format(self.name))
        self.save_filepath_step_tmp = os.path.join(self.save_dirpath_tmp, '{}'.format(self.name))

    def clean_cache(self):
        for name, step in self.all_steps.items():
            step._clean_cache()

    def _clean_cache(self):
        if os.path.exists(self.save_filepath_step_tmp):
            os.remove(self.save_filepath_step_tmp)

    @property
    def named_steps(self):
        return {step.name: step for step in self.input_steps}

    def get_step(self, name):
        return self.all_steps[name]

    @property
    def transformer_is_cached(self):
        if isinstance(self.transformer, Step):
            self._copy_transformer(self.transformer, self.name, self.cache_dirpath)
        return os.path.exists(self.cache_filepath_step_transformer)

    @property
    def output_is_cached(self):
        return os.path.exists(self.save_filepath_step_tmp)

    @property
    def output_is_saved(self):
        return os.path.exists(self.save_filepath_step_output)

    def fit_transform(self, data):
        if self.output_is_cached and not self.force_fitting:
            logger.info('step {} loading output...'.format(self.name))
            step_output_data = self._load_output(self.save_filepath_step_tmp)
        elif self.output_is_saved and self.load_saved_output and not self.force_fitting:
            logger.info('step {} loading output...'.format(self.name))
            step_output_data = self._load_output(self.save_filepath_step_output)
        else:
            step_inputs = {}
            if self.input_data is not None:
                for input_data_part in self.input_data:
                    step_inputs[input_data_part] = data[input_data_part]

            for input_step in self.input_steps:
                step_inputs[input_step.name] = input_step.fit_transform(data)

            if self.adapter:
                step_inputs = self._adapt(step_inputs)
            else:
                step_inputs = self._unpack(step_inputs)
            step_output_data = self._cached_fit_transform(step_inputs)
        return step_output_data

    def _cached_fit_transform(self, step_inputs):
        if self.transformer_is_cached and not self.force_fitting:
            logger.info('step {} loading transformer...'.format(self.name))
            self.transformer.load(self.cache_filepath_step_transformer)
            logger.info('step {} transforming...'.format(self.name))
            step_output_data = self.transformer.transform(**step_inputs)
        else:
            logger.info('step {} fitting and transforming...'.format(self.name))
            step_output_data = self.transformer.fit_transform(**step_inputs)
            logger.info('step {} saving transformer...'.format(self.name))
            self.transformer.save(self.cache_filepath_step_transformer)

        if self.cache_output:
            logger.info('step {} caching outputs...'.format(self.name))
            self._save_output(step_output_data, self.save_filepath_step_tmp)
        if self.save_output:
            logger.info('step {} saving outputs...'.format(self.name))
            self._save_output(step_output_data, self.save_filepath_step_output)
        return step_output_data

    def _load_output(self, filepath):
        return joblib.load(filepath)

    def _save_output(self, output_data, filepath):
        joblib.dump(output_data, filepath)

    def transform(self, data):
        if self.output_is_cached:
            logger.info('step {} loading output...'.format(self.name))
            step_output_data = self._load_output(self.save_filepath_step_tmp)
        elif self.output_is_saved and self.load_saved_output:
            logger.info('step {} loading output...'.format(self.name))
            step_output_data = self._load_output(self.save_filepath_step_output)
        else:
            step_inputs = {}
            if self.input_data is not None:
                for input_data_part in self.input_data:
                    step_inputs[input_data_part] = data[input_data_part]

            for input_step in self.input_steps:
                step_inputs[input_step.name] = input_step.transform(data)

            if self.adapter:
                step_inputs = self._adapt(step_inputs)
            else:
                step_inputs = self._unpack(step_inputs)
            step_output_data = self._cached_transform(step_inputs)
        return step_output_data

    def _cached_transform(self, step_inputs):
        if self.transformer_is_cached:
            logger.info('step {} loading transformer...'.format(self.name))
            self.transformer.load(self.cache_filepath_step_transformer)
            logger.info('step {} transforming...'.format(self.name))
            step_output_data = self.transformer.transform(**step_inputs)
        else:
            raise ValueError('No transformer cached {}'.format(self.name))
        if self.cache_output:
            logger.info('step {} caching outputs...'.format(self.name))
            self._save_output(step_output_data, self.save_filepath_step_tmp)
        if self.save_output:
            logger.info('step {} saving outputs...'.format(self.name))
            self._save_output(step_output_data, self.save_filepath_step_output)
        return step_output_data

    def _adapt(self, step_inputs):
        logger.info('step {} adapting inputs'.format(self.name))
        adapted_steps = {}
        for adapted_name, recipe in self.adapter.items():
            try:
                adapted_steps[adapted_name] = self._adapt_one_name(step_inputs, recipe)
            except (KeyError, ValueError) as e:
                msg = "Error in step '{}' while adapting '{}'".format(self.name, adapted_name)
                raise StepsError(msg) from e

        return adapted_steps

    def _unpack(self, step_inputs):
        logger.info('step {} unpacking inputs'.format(self.name))
        unpacked_steps = {}
        key_to_step_names = defaultdict(list)
        for step_name, step_dict in step_inputs.items():
            unpacked_steps.update(step_dict)
            for key in step_dict.keys():
                key_to_step_names[key].append(step_name)

        repeated_keys = [(key, step_names) for key, step_names in key_to_step_names.items()
                         if len(step_names) > 1]
        if len(repeated_keys) == 0:
            return unpacked_steps
        else:
            msg = "Could not unpack inputs. Following keys are present in multiple input steps:\n"\
                "\n".join(["  '{}' present in steps {}".format(key, step_names)
                           for key, step_names in repeated_keys])
            raise StepsError(msg)

    @property
    def all_steps(self):
        all_steps = {}
        all_steps = self._get_steps(all_steps)
        return all_steps

    def _adapt_one_name(self, step_inputs, recipe):
        if isinstance(recipe, tuple) and len(recipe) == 2 and not callable(recipe[1]):
            input_name, key = recipe
            return self._extract_one_item(step_inputs, input_name, key)

        if isinstance(recipe, tuple) and len(recipe) == 2 and callable(recipe[1]):
            lst, fun = recipe
        elif isinstance(recipe, list):
            lst = recipe

            def fun(x): return x
        else:
            msg = "Invalid adapting recipe: '{}'".format(recipe)
            raise ValueError(msg)

        extracted = [self._extract_one_item(step_inputs, input_name, key)
                     for input_name, key in lst]
        return fun(extracted)

    def _extract_one_item(self, step_inputs, input_name, key):
        try:
            input_dict = step_inputs[input_name]
            try:
                return input_dict[key]
            except KeyError:
                msg = "Step '{}' didn't have '{}' in its output.".format(input_name, key)
                raise StepsError(msg)
        except KeyError:
            msg = "Step '{}' doesn't have '{}' as its input step.".format(self.name, input_name)
            raise StepsError(msg)

    def _get_steps(self, all_steps):
        for input_step in self.input_steps:
            all_steps = input_step._get_steps(all_steps)
        all_steps[self.name] = self
        return all_steps

    @property
    def graph_info(self):
        graph_info = {'edges': set(),
                      'nodes': set()}

        graph_info = self._get_graph_info(graph_info)

        return graph_info

    def _get_graph_info(self, graph_info):
        for input_step in self.input_steps:
            graph_info = input_step._get_graph_info(graph_info)
            graph_info['edges'].add((input_step.name, self.name))
        graph_info['nodes'].add(self.name)
        for input_data in self.input_data:
            graph_info['nodes'].add(input_data)
            graph_info['edges'].add((input_data, self.name))
        return graph_info

    def plot_graph(self, filepath):
        plot_graph(self.graph_info, filepath)

    def __str__(self):
        return pprint.pformat(self.graph_info)

    def _repr_html_(self):
        return view_graph(self.graph_info)


class BaseTransformer:
    def fit(self, *args, **kwargs):
        return self

    def transform(self, *args, **kwargs):
        return NotImplementedError

    def fit_transform(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


class MockTransformer(BaseTransformer):
    def fit(self, *args, **kwargs):
        return self

    def transform(self, *args, **kwargs):
        return

    def fit_transform(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)


class Dummy(BaseTransformer):
    def transform(self, **kwargs):
        return kwargs


class StepsError(Exception):
    pass
