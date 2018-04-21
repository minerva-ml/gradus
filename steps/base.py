import os
import pprint
import shutil

from sklearn.externals import joblib

from .adapters import take_first_inputs
from .utils import view_graph, plot_graph, get_logger, initialize_logger

initialize_logger()
logger = get_logger()


class Step:
    def __init__(self, name, transformer, input_steps=[], input_data=[], adapter=None,
                 cache_dirpath=None, cache_output=False, save_output=False, load_saved_output=False,
                 save_graph=False, force_fitting=False):
        """Summary line.

        Step is an execution wrapper over transformer that enables building complex machine learning pipelines.
        It deals with situations where one wants to join multiple 
        For detailed examples go to the notebooks section.

        Args:
            name (int): Step name. Each step in a pipeline needs to have a unique name.
                Transformers, and Step outputs will be persisted/cached/saved under this exact name.
            transformer (obj): Step instance or object that inherits from BaseTransformer.
                When Step instance is passed transformer from that Step will be copied and used to perform transformations.
                It is useful when both train and valid data are passed in one pipeline (common situation in deep learnining).
            input_steps (list): list of Step instances default []. Current step will combine outputs from input_steps and input_data
                and pass to the transformer methods fit_transform and transform.
            input_data (list): list of str default []. Elements of this list are keys in the data dictionary that is passed
                to the pipeline/step fit_transform and/or transform methods.Current step will combine input_data and outputs from input_steps
                and pass to the transformer methods fit_transform and transform.
                Example:
                    data = {'input_1':{'X':X,
                                       'y':y}
                                       },
                            'input_2': {'X':X,
                                       'y':y}
                                       }
                           }
                    step_1 = Step(...,
                                  input_data = ['input_1']
                                  ...
                                  )
            adapter (dict): dictionary of mappings used to adapt input_steps outputs and input_data to match transform and fit_transform
                arguments for the transformer specified in this step. For each argument one needs to specify the
                argument: ([(step_name, output_key)...(input_name, output_key)], aggregation_function).
                If no aggregation_function is specified adapters.take_first_inputs function is used.
                Number of aggregation functions are available in the steps.adapters module.
                Example:
                    from steps.adapters import hstack_inputs
                    data = {'input_1':{'X':X,
                                       'y':y}
                                       },
                            'input_2': {'X':X,
                                       'y':y}
                                       }
                           }
                     step_1 = Step(name='step_1',
                                   ...
                                   )
                     step_2 = Step(name='step_2',
                                   ...
                                   )
                     step_3 = Step(name='step_3',
                                   input_steps=[step_1, step_2],
                                   input_data=['input_2'],
                                   adapter = {'X':([('step_1','X_transformed'),
                                                    ('step_2','X_transformed'),
                                                    ('step_2','categorical_features'),
                                                    ('input_2','auxilary_features'),
                                                   ], hstack_inputs)
                                              'y':(['input_1', 'y'])
                                             }
                                   ...
                                   )
                cache_dirpath (str): path to the directory where all transformers, step outputs and temporary files
                    should be stored.
                    The following subfolders will be created if they were not created by other steps:
                        transformers: transformer objects are persisted in this folder
                        outputs: step output dictionaries are persisted in this folder (if save_output=True)
                        tmp: step output dictionaries are persisted in this folder (if cache_output=True).
                            This folder is temporary and should be cleaned before/after every run
                cache_output (bool): default False. If true then step output dictionary will be cached to cache_dirpath/tmp/name after transform method
                    of the step transformer is completed. If the same step is used multiple times in the pipeline only the first time
                    the transform method is executed and later the output dictionary is loaded from the cache_dirpath/tmp/name directory.
                    Warning:
                        One should always run pipeline.clean_cache() before executing pipeline.fit_transform(data) or pipeline.transform(data)
                        Caution when working with large datasets is advised.
                save_output (bool): default False. If True then step output dictionary will be saved to cache_dirpath/outputs/name after transform method
                    of the step transformer is completed. It will save the output after every run of the step.transformer.transform method.
                    It will not be loaded unless specified with load_saved_output. It is especially useful when debugging and working with
                    ensemble models or time consuming feature extraction. One can easily persist already computed pieces of the pipeline
                    and not waste time recalculating them in the future.
                    Warning:
                        Caution when working with large datasets is advised.
                load_saved_output (bool): default False. If True then step output dictionary saved to the cache_dirpath/tmp/name will be loaded when
                    step is called.
                    Warning:
                        Reruning the same pipeline on new data with load_saved_output may lead to errors when outputs from
                        old data are loaded while user would expect the pipeline to use new data instead.
                force_fitting (bool): default False. If True then step transformer will be fitted (via fit_transform) even if
                    cache_dirpath/transformers/name exists. This is helpful when one wants to use save_output=True and load save_output=True
                    on a previous step and fit current step multiple times. That is a typical usecase when tuning hyperparameters
                    for an ensemble model trained on the outputs from first level models or a model build on features that are
                    time consuming to compute.
                save_graph (bool): default False. If true then the pipeline graph will be saved to the cache_dirpath/name_graph.json file
        """
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
                step_inputs = self.adapt(step_inputs)
            else:
                step_inputs = self.unpack(step_inputs)
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
                step_inputs = self.adapt(step_inputs)
            else:
                step_inputs = self.unpack(step_inputs)
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

    def adapt(self, step_inputs):
        logger.info('step {} adapting inputs'.format(self.name))
        adapted_steps = {}
        for adapted_name, mapping in self.adapter.items():
            if isinstance(mapping, str):
                adapted_steps[adapted_name] = step_inputs[mapping]
            else:
                if len(mapping) == 2:
                    (step_mapping, func) = mapping
                elif len(mapping) == 1:
                    step_mapping = mapping
                    func = take_first_inputs
                else:
                    raise ValueError('wrong mapping specified')

                raw_inputs = [step_inputs[step_name][step_var] for step_name, step_var in step_mapping]
                adapted_steps[adapted_name] = func(raw_inputs)
        return adapted_steps

    def unpack(self, step_inputs):
        logger.info('step {} unpacking inputs'.format(self.name))
        unpacked_steps = {}
        for step_name, step_dict in step_inputs.items():
            unpacked_steps = {**unpacked_steps, **step_dict}
        return unpacked_steps

    @property
    def all_steps(self):
        all_steps = {}
        all_steps = self._get_steps(all_steps)
        return all_steps

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
