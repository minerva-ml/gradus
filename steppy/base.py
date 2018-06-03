import glob
import os
import pprint
import shutil
from collections import defaultdict

from sklearn.externals import joblib

from steppy.adapter import Adapter, AdapterError
from steppy.utils import display_pipeline, save_as_png, get_logger, initialize_logger

initialize_logger()
logger = get_logger()


class Step:
    """Building block of steppy pipelines.

    Step is an execution wrapper over the transformer (see BaseTransformer) that enables building complex machine learning pipelines.
    With Step you can:
        1. design multiple input/output data flows
        2. handle persistence/caching of both models (transformers) and intermediate results.
    Step executes fit_transform on every step recursively starting from the very last step and making its way forward
    through the input_steps. If step transformer was fitted already then said transformer is loaded in and the transform method
    is executed.
    One can easily debug the data flow by plotting the pipeline graph with either step.save_as_png(filepath) method
    or simply returning it in a jupyter notebook cell.
    Every part of the pipeline can be easily accessed via step.get_step(name) method which makes it easy to reuse parts of the pipeline
    across multiple solutions.
    """
    def __init__(self,
                 name,
                 transformer,
                 experiment_directory=None,
                 input_data=None,
                 input_steps=None,
                 adapter=None,
                 cache_output=False,
                 save_output=False,
                 load_saved_output=False,
                 save_upstream_pipeline_structure=False,
                 force_fitting=False):
        """
        Args:
            name (str): Step name. Each step in a pipeline needs to have a unique name.
                Transformers, and Step outputs will be saved under this name.
            transformer (obj): Step instance or object that inherits from BaseTransformer.
                When Step instance is passed, transformer from that Step will be copied and used to perform transformations.
                It is useful when both train and valid data are passed in one pipeline (common situation in deep learning).
            experiment_directory (str): path to the directory where all execution artifacts will be stored.
                The following sub-directories will be created, if they were not created by other Steps:
                    transformers: transformer objects are persisted in this folder
                    outputs: step output dictionaries are persisted in this folder (if save_output=True)
                    tmp: step output dictionaries are persisted in this folder (if cache_output=True).
                        This folder is temporary and should be cleaned before every run.
            input_steps (list): list of Step instances default []. Current step will combine outputs from input_steps and input_data
                and pass to the transformer methods fit_transform and transform.
            input_data (list): list of str default []. Elements of this list are keys in the data dictionary that is passed
                to the Step's fit_transform or transform methods. Current step will combine input_data and outputs from input_steps
                and pass to the transformer methods fit_transform and transform.
                Example:
                    data = {'input_1': {'X': X,
                                        'y': y}
                                        },
                            'input_2': {'X': X,
                                        'y': y}
                                        }
                            }
                    step_1 = Step(...,
                                  input_data = ['input_1']
                                  ...
                                  )
            adapter (obj): Must be an instance of the steppy.adapter.Adapter class.

            cache_output (bool): default False. If true then step output dictionary will be cached to exp_dir/tmp/name after transform method
                of the step transformer is completed. If the same step is used multiple times in the pipeline only the first time
                the transform method is executed and later the output dictionary is loaded from the exp_dir/tmp/name directory.
                Warning:
                    One should always run pipeline.clean_cache() before executing pipeline.fit_transform(data) or pipeline.transform(data)
                    Caution when working with large datasets is advised.
            save_output (bool): default False. If True then Step output dictionary will be saved to
                `exp_dir/outputs/name` after transform method
                of the step transformer is completed. It will save the output after every run of the step.transformer.transform method.
                It will not be loaded unless specified with load_saved_output. It is especially useful when debugging and working with
                ensemble models or time consuming feature extraction. One can easily persist already computed pieces of the pipeline
                and not waste time recalculating them in the future.
                Warning:
                    Caution when working with large datasets is advised.
            load_saved_output (bool): default False. If True then step output dictionary saved to the exp_dir/tmp/name will be loaded when
                step is called.
                Warning:
                    Reruning the same pipeline on new data with load_saved_output may lead to errors when outputs from
                    old data are loaded while user would expect the pipeline to use new data instead.
            force_fitting (bool): default False. If True then step transformer will be fitted (via fit_transform) even if
                exp_dir/transformers/name exists. This is helpful when one wants to use save_output=True and load save_output=True
                on a previous step and fit current step multiple times. That is a typical usecase when tuning hyperparameters
                for an ensemble model trained on the outputs from first level models or a model build on features that are
                time consuming to compute.
            save_upstream_pipeline_structure (bool): default False.
                If true then the upstream pipeline structure (with regard to the current Step) will be saved as json
                in the exp_dir
        """
        assert isinstance(name, str), 'name must be str'
        assert isinstance(experiment_directory, str), 'exp_dir (experiment_directory) must be str'
        if adapter is not None:
            assert isinstance(adapter, Adapter), 'adapter must be an instance of {}'.format(str(Adapter))

        logger.info('initializing step {}'.format(name))

        self.exp_dir = os.path.join(experiment_directory)
        self._prepare_experiment_directories()

        self.name = name
        self.transformer = transformer

        self.input_steps = input_steps or []
        self.input_data = input_data or []
        self.adapter = adapter

        self.cache_output = cache_output
        self.save_output = save_output
        self.load_saved_output = load_saved_output
        self.force_fitting = force_fitting

        if save_upstream_pipeline_structure:
            save_dir = os.path.join(self.exp_dir, '{}_upstream_structure.json'.format(self.name))
            logger.info('saving upstream pipeline structure to {}'.format(save_dir))
            joblib.dump(self.upstream_pipeline_structure, save_dir)

    @property
    def upstream_pipeline_structure(self):
        """build dictionary with entire upstream pipeline structure (with regard to the current Step).

        Returns:
            dict: dictionary describing the upstream pipeline structure. It has two keys: 'edges' and 'nodes', where:
                value of 'edges' is set of tuples (input_step.name, self.name)
                value of 'nodes' is set of all step names upstream to this Step
        """
        structure_dict = {'edges': set(),
                          'nodes': set()}
        structure_dict = self._build_structure_dict(structure_dict)
        return structure_dict

    @property
    def all_steps(self):
        """build dictionary with all Step instances that are upstream to self.

        Returns:
            all_steps (dict): dictionary where keys are Step names (str) and values are Step instances (obj)
        """
        all_steps = {}
        all_steps = self._get_steps(all_steps)
        return all_steps

    @property
    def transformer_is_cached(self):
        """(bool): True if transformer exists under the directory self.exp_dir_transformers_step
        """
        if isinstance(self.transformer, Step):
            self._copy_transformer(self.transformer, self.name, self.exp_dir)
        return os.path.exists(self.exp_dir_transformers_step)

    @property
    def output_is_cached(self):
        """(bool): True if step outputs exists under the exp_dir/tmp/name.
            See cache_output.
        """
        return os.path.exists(self.exp_dir_tmp_step)

    @property
    def output_is_saved(self):
        """(bool): True if step outputs exists under the exp_dir/outputs/name.
            See save_output.
        """
        return os.path.exists(self.exp_dir_outputs_step)

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
            step_output_data = self._load_output(self.exp_dir_tmp_step)
        elif self.output_is_saved and self.load_saved_output and not self.force_fitting:
            logger.info('step {} loading output...'.format(self.name))
            step_output_data = self._load_output(self.exp_dir_outputs_step)
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
            logger.info('step {} loading cached output...'.format(self.name))
            step_output_data = self._load_output(self.exp_dir_tmp_step)
        elif self.output_is_saved and self.load_saved_output:
            logger.info('step {} loading output...'.format(self.name))
            step_output_data = self._load_output(self.exp_dir_outputs_step)
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
        """Removes everything from the directory 'self.exp_dir_tmp'
        """
        logger.info('cleaning cache...')
        paths = glob.glob(os.path.join(self.exp_dir_tmp, '*'))
        for path in paths:
            logger.info('removing {}'.format(path))
            os.remove(path)
        logger.info('cleaning cache done')

    def get_step(self, name):
        """Extracts step by name from the pipeline.

        Extracted step is a fully functional pipeline as well.
        This method can be used to port parts of the pipeline between problems.

        Args:
            name (str): name of the step to be fetched
        Returns:
            Step (obj): extracted step
        """
        return self.all_steps[name]

    def save_pipeline_as_png(self, filepath):
        """Creates pipeline graph and saves it as png

        Pydot graph is created and saved to filepath as png image. This feature is useful for debugging purposes
        especially when working with complex pipelines.

        Args:
            filepath (str): filepath to which the png with pipeline visualization should be saved
        """
        assert isinstance(filepath, str), 'Step {} error, filepath must be str. Got {} instead'.format(self.name,
                                                                                                       type(filepath))
        save_as_png(self.upstream_pipeline_structure, filepath)

    def _copy_transformer(self, step, name, dirpath):
        self.transformer = self.transformer.transformer

        original_filepath = os.path.join(step.exp_dir, 'transformers', step.name)
        copy_filepath = os.path.join(dirpath, 'transformers', name)
        logger.info('copying transformer from {} to {}'.format(original_filepath, copy_filepath))
        shutil.copyfile(original_filepath, copy_filepath)

    def _prepare_experiment_directories(self):
        logger.info('initializing experiment directories under {}'.format(self.exp_dir))

        for dir_name in ['transformers', 'outputs', 'tmp']:
            os.makedirs(os.path.join(self.exp_dir, dir_name), exist_ok=True)

        self.exp_dir_transformers = os.path.join(self.exp_dir, 'transformers')
        self.exp_dir_outputs = os.path.join(self.exp_dir, 'outputs')
        self.exp_dir_tmp = os.path.join(self.exp_dir, 'tmp')

        self.exp_dir_transformers_step = os.path.join(self.exp_dir_transformers, self.name)
        self.exp_dir_outputs_step = os.path.join(self.exp_dir_outputs, '{}'.format(self.name))
        self.exp_dir_tmp_step = os.path.join(self.exp_dir_tmp, '{}'.format(self.name))

    def _cached_fit_transform(self, step_inputs):
        if self.transformer_is_cached and not self.force_fitting:
            logger.info('step {} loading transformer...'.format(self.name))
            self.transformer.load(self.exp_dir_transformers_step)
            logger.info('step {} transforming...'.format(self.name))
            step_output_data = self.transformer.transform(**step_inputs)
        else:
            logger.info('step {} fitting and transforming...'.format(self.name))
            step_output_data = self.transformer.fit_transform(**step_inputs)
            logger.info('step {} saving transformer...'.format(self.name))
            self.transformer.save(self.exp_dir_transformers_step)

        if self.cache_output:
            logger.info('step {} caching outputs...'.format(self.name))
            self._save_output(step_output_data, self.exp_dir_tmp_step)
        if self.save_output:
            logger.info('step {} saving outputs...'.format(self.name))
            self._save_output(step_output_data, self.exp_dir_outputs_step)
        return step_output_data

    def _load_output(self, filepath):
        logger.info('loading output from {}'.format(filepath))
        return joblib.load(filepath)

    def _save_output(self, output_data, filepath):
        logger.info('saving output to {}'.format(filepath))
        joblib.dump(output_data, filepath)

    def _cached_transform(self, step_inputs):
        if self.transformer_is_cached:
            logger.info('step {} loading transformer...'.format(self.name))
            self.transformer.load(self.exp_dir_transformers_step)
            logger.info('step {} transforming...'.format(self.name))
            step_output_data = self.transformer.transform(**step_inputs)
        else:
            raise ValueError('No transformer cached {}'.format(self.name))
        if self.cache_output:
            logger.info('step {} caching outputs...'.format(self.name))
            self._save_output(step_output_data, self.exp_dir_tmp_step)
        if self.save_output:
            logger.info('step {} saving outputs...'.format(self.name))
            self._save_output(step_output_data, self.exp_dir_outputs_step)
        return step_output_data

    def _adapt(self, step_inputs):
        logger.info('step {} adapting inputs...'.format(self.name))
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

    def _build_structure_dict(self, structure_dict):
        for input_step in self.input_steps:
            structure_dict = input_step._build_structure_dict(structure_dict)
            structure_dict['edges'].add((input_step.name, self.name))
        structure_dict['nodes'].add(self.name)
        for input_data in self.input_data:
            structure_dict['nodes'].add(input_data)
            structure_dict['edges'].add((input_data, self.name))
        return structure_dict

    def _repr_html_(self):
        return display_pipeline(self.upstream_pipeline_structure)

    def __str__(self):
        return pprint.pformat(self.upstream_pipeline_structure)


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

        Specific implementation of model parameter persistence should be implemented here.
        In case of transformers that do not learn any parameters one can leave this method as is.

        Args:
            filepath (str): filepath where the transformer parameters should be saved
        """
        joblib.dump({}, filepath)


class IdentityOperation(BaseTransformer):
    """Transformer that performs identity operation, f(x)=x.
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
