import os
import pprint
import shutil
from collections import defaultdict

from sklearn.externals import joblib

from steppy.adapter import AdapterError
from steppy.utils import view_graph, save_graph, get_logger, initialize_logger

initialize_logger()
logger = get_logger()


class Step:
    """Building block of steps pipelines.

    Step is an execution wrapper over the transformer (see BaseTransformer class documentation) that enables building complex machine learning pipelines.
    It handles multiple input/output data flows, has build-in persistence/caching of both models (transformers) and
    intermediate step outputs.
    Step executes fit_transform on every step recursively starting from the very last step and making its way forward
    through the input_steps. If step transformer was fitted already then said transformer is loaded in and the transform method
    is executed.
    One can easily debug the data flow by plotting the pipeline graph with either step.save_graph(filepath) method
    or simply returning it in a jupyter notebook cell.
    Every part of the pipeline can be easily accessed via step.get_step(name) method which makes it easy to reuse parts of the pipeline
    across multiple solutions.
    """
    def __init__(self,
                 name,
                 transformer,
                 input_steps=None,
                 input_data=None,
                 adapter=None,
                 cache_dirpath=None,
                 cache_output=False,
                 save_output=False,
                 load_saved_output=False,
                 save_graph=False,
                 force_fitting=False):
        """
        Args:
            name (str): Step name. Each step in a pipeline needs to have a unique name.
                Transformers, and Step outputs will be persisted/cached/saved under this exact name.
            transformer (obj): Step instance or object that inherits from BaseTransformer.
                When Step instance is passed transformer from that Step will be copied and used to perform transformations.
                It is useful when both train and valid data are passed in one pipeline (common situation in deep learning).
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
                                       'y':y
                                       },
                            'input_2': {'X':X,
                                       'y':y
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

        self.input_steps = input_steps or []
        self.input_data = input_data or []
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

    @property
    def graph_info(self):
        """(dict): dictionary describing the pipeline execution graph.
        """
        graph_info = {'edges': set(),
                      'nodes': set()}

        graph_info = self._get_graph_info(graph_info)

        return graph_info

    @property
    def all_steps(self):
        """(dict): dictionary of steps in the pipeline.
        """
        all_steps = {}
        all_steps = self._get_steps(all_steps)
        return all_steps

    @property
    def transformer_is_cached(self):
        """(bool): True if transformer exists under the cache_dirpath/transformers/name
        """
        if isinstance(self.transformer, Step):
            self._copy_transformer(self.transformer, self.name, self.cache_dirpath)
        return os.path.exists(self.cache_filepath_step_transformer)

    @property
    def output_is_cached(self):
        """(bool): True if step outputs exists under the cache_dirpath/tmp/name.
            See cache_output.
        """
        return os.path.exists(self.save_filepath_step_tmp)

    @property
    def output_is_saved(self):
        """(bool): True if step outputs exists under the cache_dirpath/outputs/name.
            See save_output.
        """
        return os.path.exists(self.save_filepath_step_output)

    def fit_transform(self, data):
        """fits the model and transforms data or loads already processed data

        Loads cached/saved outputs or adapts data for the current transformer and executes transformer.fit_transform

        Args:
            data (dict): data dictionary with keys as input names and values as dictionaries of key:value pairs that can
                be passed to the step.transformer.fit_transform method
                Example:
                    data = {'input_1':{'X':X,
                                       'y':y
                                       },
                            'input_2': {'X':X,
                                       'y':y
                                       }
                           }
        Returns:
            dict: step outputs from the transformer.fit_transform method
        """
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

    def transform(self, data):
        """transforms data or loads already processed data

        Loads cached/saved outputs or adapts data for the current transformer and executes transformer.transform

        Args:
            data (dict): data dictionary with keys as input names and values as dictionaries of key:value pairs that can
                be passed to the step.transformer.fit_transform method
                Example:
                    data = {'input_1':{'X':X,
                                       'y':y
                                       },
                            'input_2': {'X':X,
                                       'y':y
                                       }
                           }
        Returns:
            dict: step outputs from the transformer.transform method
        """
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

    def clean_cache(self):
        """Removes cached step outputs.

        It removes all cached step output from the cache_dirpath/tmp
        """
        for name, step in self.all_steps.items():
            step._clean_cache()

    def get_step(self, name):
        """Extracts step by name from the pipeline.

        Extracted step is a fully functional pipeline as well.
        This method can be used to port parts of the pipeline between problems.

        Args:
            name (str): name of the step to be fetched
        Returns:
            Step: extracted step
        """
        return self.all_steps[name]

    def save_graph(self, filepath):
        """Creates pipeline graph and saves it to a file

        Pydot graph is created and saved to filepath. This feature is usefull for debugging purposes especially
        when working with complex pipelines.

        Args:
            filepath (str): filepath where the graph should be saved
        """
        save_graph(self.graph_info, filepath)

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

    def _clean_cache(self):
        if os.path.exists(self.save_filepath_step_tmp):
            os.remove(self.save_filepath_step_tmp)

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
        try:
            return self.adapter.adapt(step_inputs)
        except AdapterError as e:
            msg = "Error while adapting step '{}'".format(self.name)
            raise StepsError(msg) from e

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

    def _get_steps(self, all_steps):
        for input_step in self.input_steps:
            all_steps = input_step._get_steps(all_steps)
        all_steps[self.name] = self
        return all_steps

    def _get_graph_info(self, graph_info):
        for input_step in self.input_steps:
            graph_info = input_step._get_graph_info(graph_info)
            graph_info['edges'].add((input_step.name, self.name))
        graph_info['nodes'].add(self.name)
        for input_data in self.input_data:
            graph_info['nodes'].add(input_data)
            graph_info['edges'].add((input_data, self.name))
        return graph_info

    def __str__(self):
        return pprint.pformat(self.graph_info)

    def _repr_html_(self):
        return view_graph(self.graph_info)


class BaseTransformer:
    """Abstraction on two level fit and transform execution.

    Base transformer is an abstraction strongly inspired by the sklearn.Transformer sklearn.Estimator.
    Two main concepts are:
        1. Every action that can be performed on data (transformation, model training) can be performed in two steps
        fitting (where trainable parameters are estimated) and transforming (where previously estimated parameters are used
        to transform the data into desired state)
        2. Every transformer knows how it should be saved and loaded (especially useful when working with Keras/Pytorch and Sklearn)
        in one pipeline
    """

    def __init__(self):
        self.estimator = None

    def fit(self, *args, **kwargs):
        """Performs estimation of trainable parameters

        All model estimations with sklearn, keras, pytorch models as well as some preprocessing techniques (normalization)
        estimate parameters based on data (training data). Those parameters are trained during fit execution and
        are persisted for the future.
        Only the estimation logic, nothing else.

        Args:
            args: positional arguments (can be anything)
            kwargs: keyword arguments (can be anything)

        Returns:
            BaseTransformer: self object
        """
        return self

    def transform(self, *args, **kwargs):
        """Performs transformation of data

        All data transformation including prediction with deep learning/machine learning models can be performed here.
        No parameters should be estimated in this method nor stored as class attributes.
        Only the transformation logic, nothing else.

        Args:
            args: positional arguments (can be anything)
            kwargs: keyword arguments (can be anything)

        Returns:
            dict: outputs
        """
        raise NotImplementedError

    def fit_transform(self, *args, **kwargs):
        """Performs fit followed by transform

        This method simply combines fit and transform.

        Args:
            args: positional arguments (can be anything)
            kwargs: keyword arguments (can be anything)

        Returns:
            dict: outputs
        """
        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)

    def load(self, filepath):
        """Loads the trainable parameters of the transformer

        Specific implementation of loading persisted model parameters should be implemented here.
        In case of transformers that do not learn any parameters one can leave this method as is.

        Args:
            filepath (str): filepath from which the transformer should be loaded
        Returns:
            BaseTransformer: self instance
        """
        return self

    def save(self, filepath):
        """Saves the trainable parameters of the transformer

        Specific implementation of model parameter persistance should be implemented here.
        In case of transformers that do not learn any parameters one can leave this method as is.

        Args:
            filepath (str): filepath where the transformer parameters should be saved
        """
        joblib.dump({}, filepath)


class IdentityOperation(BaseTransformer):
    """Transformer that performs identity operation, f(x)=x.

    It is sometimes useful to organize the outputs from previous steps, join them together or rename them before
    passing to the next step. Typical use-case would be to join features extracted with
    multiple transformers into one object called joined_features. In that case the adapter attribute is used to define
    the mapping/joining scheme.

    """
    def transform(self, **kwargs):
        return kwargs


class StepsError(Exception):
    pass


def make_transformer(func):
    class StaticTransformer(BaseTransformer):
        def transform(self, *args, **kwargs):
            return func(*args, **kwargs)
    return StaticTransformer()
