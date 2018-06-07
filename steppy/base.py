import glob
import os
import pprint
import shutil
from collections import defaultdict

from sklearn.externals import joblib

from steppy.adapter import Adapter, AdapterError
from steppy.utils import display_pipeline, persist_as_png, get_logger, initialize_logger

initialize_logger()
logger = get_logger()


class Step:
    """Step is a building block of steppy pipelines.

    It is an execution wrapper over the transformer (see :class:`~steppy.base.BaseTransformer`),
    which realizes single operation on data. With Step you can:

    1. design multiple input/output data flows and connections between Steps.
    2. handle persistence and caching of transformers and intermediate results.

    Step executes `fit_transform` method inspired by the sklearn on every step recursively
    starting from the very last Step and making its way forward through the `input_steps`.
    One can easily debug the data flow by plotting the pipeline graph
    (see: :func:`~steppy.utils.persist_as_png`) or return step in a jupyter notebook cell.

    Attributes:
        name (str): Step name.
            Each step in a pipeline must have a unique name. This names is used to persist or cache
            transformers and outputs of this Step.

        transformer (obj): object that inherits from BaseTransformer or Step instance.
            When Step instance is passed, transformer from that Step will be copied and used to
            perform transformations. It is useful when both train and valid data are passed in
            one pipeline (common situation in deep learning).

        experiment_directory (str): path to the directory where all execution artifacts will be
            stored. The following sub-directories will be created, if they were not created by
            other Steps:

            * transformers: transformer objects are persisted in this folder
            * outputs:      step output dictionaries are persisted in this folder
              (if ``persist_output=True``)
            * cache:        step output dictionaries are cached in this folder
              (if ``cache_output=True``).

        input_data (list): Elements of this list are keys in the data dictionary that is passed
            to the Step's `fit_transform` and `transform` methods.
            List of str, default is empty list.

            Example:

                .. code-block:: python

                    data_train = {'input': {'images': X_train,
                                            'labels': y_train}
                                 }

                    my_step = Step(name='random_forest',
                                   transformer=RandomForestTransformer(),
                                   input_data=['input']
                                   )

                    my_step.fit_transform(data_train)

                `data_train` is dictionary where:

                 * keys are names of data packets,
                 * values are data packets, that is dictionaries that describes dataset.
                   In this example keys in the data packet are `images` and `labels` and values
                   are actual data of any type.

                `Step.input_data` takes the key from `data_train` (values must match!) and extracts
                actual data that will be passed to the `fit_transform` and `transform` method of
                the `self.transformer`.

        input_steps (list): List of input Steps that the current Step uses as its input.
            list of Step instances, default is empty list.
            Current Step will combine outputs from `input_steps` and `input_data` using `adapter`.
            Then pass it to the transformer methods `fit_transform` and `transform`.

            Example:

                .. code-block:: python

                    self.input_steps=[cnn_step, rf_step, ensemble_step, guesses_step]

                Each element of the list is Step instance.

        adapter (obj): It renames and arranges inputs that are passed to the Transformer
            (see :class:`~steppy.base.BaseTransformer`).
            Default is ``None``.
            If ``not None``, then must be an instance of the :class:`~steppy.adapter.Adapter` class.

            Example:
                .. code-block:: python

                    self.adapter=Adapter({'X': E('input', 'images'),
                                          'y': E('input', 'labels')}
                                         )

            Adapter simplifies the renaming and combining of inputs from multiple steps.
            In this example, after the adaptation:

            * `X` is key to the data stored under the `images` key
            * `y` is key to the data stored under the `labels` key

                where both `images` and `labels` keys comes from `input`
                (see :attr:`~steppy.base.Step.input_data`)

        cache_output (bool): If True, Step output dictionary will be cached to the
            ``<experiment_directory>/cache/<name>``, when transform method of the Step transformer
            is completed. If the same Step is used multiple times, transform method is invoked
            only once. Further invokes simply load output from the
            ``<experiment_directory>/cache/<name>`` directory.
            Default ``False``: do not cache outputs

            Warning:
                One should always run `pipeline.clean_cache()` before executing
                `pipeline.fit_transform(data)` or `pipeline.transform(data)`
                When working with large datasets, cache might be very large.

        persist_output (bool): If True, persist Step output to disk under the
            ``<experiment_directory>/outputs/<name>`` directory.
            Default ``False``: do not persist any files to disk.
            If True then Step output dictionary will be persisted to the
            ``<experiment_directory>/outputs/<name>`` directory, after transform method of the Step
            transformer is completed. Step persists to disk the output after every run of the
            transformer's transform method. It means that Step overrides files. See also
            `load_persisted_output` parameter.

            Warning:
                When working with large datasets, cache might be very large.

        load_persisted_output (bool): If True, Step output dictionary already persisted to the
            ``<experiment_directory>/cache/<name>`` will be loaded when Step is called.
            Default ``False``: do not load persisted output.
            Useful when debugging and working with ensemble models or time consuming feature
            extraction. One can easily persist already computed pieces of the pipeline and save
            time by loading them instead of calculating.

            Warning:
                Re-running the same pipeline on new data with `load_persisted_output` set ``True``
                may lead to errors when outputs from old data are loaded while user would expect
                the pipeline to use new data instead.

        force_fitting (bool): If True, Step transformer will be fitted (via `fit_transform`)
            even if ``<experiment_directory>/transformers/<step_name>`` exists.
            Default ``False``: do not force fitting of the transformer.
            Helpful when one wants to use ``persist_output=True`` and load ``persist_output=True``
            on a previous Step and fit current Step multiple times. This is a typical scenario
            for tuning hyperparameters for an ensemble model trained on the outputs from first
            level models or a model build on features that are time consuming to compute.

        persist_upstream_pipeline_structure (bool): If True, the upstream pipeline structure
            (with regard to the current Step) will be persisted as json file in the
            ``experiment_directory``.
            Default ``False``: do not persist upstream pipeline structure.
    """
    def __init__(self,
                 name,
                 transformer,
                 experiment_directory,
                 input_data=None,
                 input_steps=None,
                 adapter=None,
                 cache_output=False,
                 persist_output=False,
                 load_persisted_output=False,
                 force_fitting=False,
                 persist_upstream_pipeline_structure=False):

        assert isinstance(name, str), 'Step name must be str, got {} instead.'.format(type(name))
        assert isinstance(experiment_directory, str), 'Step {} error, experiment_directory must ' \
            'be str, got {} instead.'.format(name, type(experiment_directory))

        if input_data is not None:
            assert isinstance(input_data, list), 'Step {} error, input_data must be list, ' \
                'got {} instead.'.format(name, type(input_data))
        if input_steps is not None:
            assert isinstance(input_steps, list), 'Step {} error, input_steps must be list, ' \
                'got {} instead.'.format(name, type(input_steps))
        if adapter is not None:
            assert isinstance(adapter, Adapter), 'Step {} error, adapter must be an instance ' \
                'of {}'.format(name, str(Adapter))

        assert isinstance(cache_output, bool), 'Step {} error, cache_output must be bool, ' \
            'got {} instead.'.format(name, type(cache_output))
        assert isinstance(persist_output, bool), 'Step {} error, persist_output must be bool, ' \
            'got {} instead.'.format(name, type(persist_output))
        assert isinstance(load_persisted_output, bool), 'Step {} error, load_persisted_output ' \
            'must be bool, got {} instead.'.format(name, type(load_persisted_output))
        assert isinstance(force_fitting, bool), 'Step {} error, force_fitting must be bool, ' \
            'got {} instead.'.format(name, type(force_fitting))
        assert isinstance(persist_upstream_pipeline_structure, bool), 'Step {} error, ' \
            'persist_upstream_pipeline_structure must be bool, got {} instead.'\
            .format(name, type(persist_upstream_pipeline_structure))

        logger.info('initializing Step {}...'.format(name))

        self.name = name
        self.transformer = transformer

        self.input_steps = input_steps or []
        self.input_data = input_data or []
        self.adapter = adapter

        self.cache_output = cache_output
        self.persist_output = persist_output
        self.load_persisted_output = load_persisted_output
        self.force_fitting = force_fitting

        self.exp_dir = os.path.join(experiment_directory)
        self._prepare_experiment_directories()

        if persist_upstream_pipeline_structure:
            persist_dir = os.path.join(self.exp_dir, '{}_upstream_structure.json'.format(self.name))
            logger.info('saving upstream pipeline structure to {}'.format(persist_dir))
            joblib.dump(self.upstream_pipeline_structure, persist_dir)

        logger.info('Step {} initialized'.format(name))

    @property
    def upstream_pipeline_structure(self):
        """Build dictionary with entire upstream pipeline structure
        (with regard to the current Step).

        Returns:
            dict: dictionary describing the upstream pipeline structure. It has two keys:
            ``'edges'`` and ``'nodes'``, where:

            - value of ``'edges'`` is set of tuples ``(input_step.name, self.name)``
            - value of ``'nodes'`` is set of all step names upstream to this Step
        """
        structure_dict = {'edges': set(),
                          'nodes': set()}
        structure_dict = self._build_structure_dict(structure_dict)
        return structure_dict

    @property
    def all_steps(self):
        """Build dictionary with all Step instances that are upstream to `self`.

        Returns:
            all_steps (dict): dictionary where keys are Step names (str) and values are Step
            instances (obj)
        """
        all_steps = {}
        all_steps = self._get_steps(all_steps)
        return all_steps

    @property
    def transformer_is_cached(self):
        """(bool): True if transformer exists under the directory
        ``<experiment_directory>/transformers/<step_name>``
        """
        if isinstance(self.transformer, Step):
            self._copy_transformer(self.transformer, self.name, self.exp_dir)
        return os.path.exists(self.exp_dir_transformers_step)

    @property
    def output_is_cached(self):
        """(bool): True if step outputs exists under the ``<experiment_directory>/cache/<name>``.
            See `cache_output`.
        """
        return os.path.exists(self.exp_dir_cache_step)

    @property
    def output_is_persisted(self):
        """(bool): True if step outputs exists under the ``<experiment_directory>/outputs/<name>``.
            See :attr:`~steppy.base.Step.persist_output`.
        """
        return os.path.exists(self.exp_dir_outputs_step)

    def fit_transform(self, data):
        """Fit the model and transform data or load already processed data.

        Loads cached or persisted outputs or adapts data for the current transformer and
        executes ``transformer.fit_transform``.

        Args:
            data (dict): data dictionary with keys as input names and values as dictionaries of
                key-value pairs that can be passed to the ``self.transformer.fit_transform`` method.
                Example:

                .. code-block:: python

                    data = {'input_1': {'X': X,
                                        'y': y},
                            'input_2': {'X': X,
                                        'y': y}
                            }

        Returns:
            dict: Step outputs from the ``self.transformer.fit_transform`` method
        """
        if self.output_is_cached and not self.force_fitting:
            logger.info('step {} loading output...'.format(self.name))
            step_output_data = self._load_output(self.exp_dir_cache_step)
        elif self.output_is_persisted and self.load_persisted_output and not self.force_fitting:
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
        """Transforms data or loads already processed data.

        Loads cached persisted outputs or adapts data for the current transformer and executes
        its `transform` method.

        Args:
            data (dict): data dictionary with keys as input names and values as dictionaries of
                key:value pairs that can be passed to the ``step.transformer.fit_transform`` method

                Example:

                    .. code-block:: python

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
            step_output_data = self._load_output(self.exp_dir_cache_step)
        elif self.output_is_persisted and self.load_persisted_output:
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
        """Removes everything from the directory ``<experiment_directory>/cache``.
        """
        logger.info('cleaning cache...')
        paths = glob.glob(os.path.join(self.exp_dir_cache, '*'))
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

    def persist_pipeline_diagram(self, filepath):
        """Creates pipeline diagram and persists it to disk as png file.

        Pydot graph is created and persisted to disk as png file under the filepath directory.

        Args:
            filepath (str): filepath to which the png with pipeline visualization should
                be persisted
        """
        assert isinstance(filepath, str), 'Step {} error, filepath must be str. Got {}' \
                                          ' instead'.format(self.name, type(filepath))
        persist_as_png(self.upstream_pipeline_structure, filepath)

    def _copy_transformer(self, step, name, dirpath):
        self.transformer = self.transformer.transformer

        original_filepath = os.path.join(step.exp_dir, 'transformers', step.name)
        copy_filepath = os.path.join(dirpath, 'transformers', name)
        logger.info('copying transformer from {} to {}'.format(original_filepath, copy_filepath))
        shutil.copyfile(original_filepath, copy_filepath)

    def _prepare_experiment_directories(self):
        logger.info('initializing experiment directories under {}'.format(self.exp_dir))

        for dir_name in ['transformers', 'outputs', 'cache']:
            os.makedirs(os.path.join(self.exp_dir, dir_name), exist_ok=True)

        self.exp_dir_transformers = os.path.join(self.exp_dir, 'transformers')
        self.exp_dir_outputs = os.path.join(self.exp_dir, 'outputs')
        self.exp_dir_cache = os.path.join(self.exp_dir, 'cache')

        self.exp_dir_transformers_step = os.path.join(self.exp_dir_transformers, self.name)
        self.exp_dir_outputs_step = os.path.join(self.exp_dir_outputs, '{}'.format(self.name))
        self.exp_dir_cache_step = os.path.join(self.exp_dir_cache, '{}'.format(self.name))

        logger.info('done: initializing experiment directories')

    def _cached_fit_transform(self, step_inputs):
        if self.transformer_is_cached and not self.force_fitting:
            logger.info('Step {}, loading transformer from the {}'
                        .format(self.name, self.exp_dir_transformers_step))
            self.transformer.load(self.exp_dir_transformers_step)
            logger.info('Step {}, transforming...'.format(self.name))
            step_output_data = self.transformer.transform(**step_inputs)
        else:
            logger.info('Step {}, fitting and transforming...'.format(self.name))
            step_output_data = self.transformer.fit_transform(**step_inputs)
            logger.info('Step {}, persisting transformer to the {}'
                        .format(self.name, self.exp_dir_transformers_step))
            self.transformer.persist(self.exp_dir_transformers_step)

        if self.cache_output:
            logger.info('Step {}, caching outputs to the {}'
                        .format(self.name, self.exp_dir_cache_step))
            self._persist_output(step_output_data, self.exp_dir_cache_step)
        if self.persist_output:
            logger.info('Step {}, persisting outputs to the {}'
                        .format(self.name, self.exp_dir_outputs_step))
            self._persist_output(step_output_data, self.exp_dir_outputs_step)
        return step_output_data

    def _load_output(self, filepath):
        logger.info('Step {}, loading output from {}'.format(self.name, filepath))
        return joblib.load(filepath)

    def _persist_output(self, output_data, filepath):
        logger.info('Step {}, persisting output to the {}'.format(self.name, filepath))
        joblib.dump(output_data, filepath)

    def _cached_transform(self, step_inputs):
        if self.transformer_is_cached:
            logger.info('Step {}, loading transformer from the {}'
                        .format(self.name, self.exp_dir_transformers_step))
            self.transformer.load(self.exp_dir_transformers_step)
            logger.info('Step {}, transforming...'.format(self.name))
            step_output_data = self.transformer.transform(**step_inputs)
        else:
            raise ValueError('No transformer cached {}'.format(self.name))
        if self.cache_output:
            logger.info('Step {}, caching outputs to the {}'
                        .format(self.name, self.exp_dir_cache_step))
            self._persist_output(step_output_data, self.exp_dir_cache_step)
        if self.persist_output:
            logger.info('Step {}, persisting outputs to the {}'
                        .format(self.name, self.exp_dir_outputs_step))
            self._persist_output(step_output_data, self.exp_dir_outputs_step)
        return step_output_data

    def _adapt(self, step_inputs):
        logger.info('Step {}, adapting inputs...'.format(self.name))
        try:
            return self.adapter.adapt(step_inputs)
        except AdapterError as e:
            msg = "Error while adapting step '{}'. Check Step inputs".format(self.name)
            raise StepsError(msg) from e

    def _unpack(self, step_inputs):
        logger.info('Step {}, unpacking inputs...'.format(self.name))
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

    Base transformer is an abstraction strongly inspired by the ``sklearn.Transformer`` and
    ``sklearn.Estimator``. Two main concepts are:

        1. Every action that can be performed on data (transformation, model training) can be
        performed in two steps: fitting (where trainable parameters are estimated) and transforming
        (where previously estimated parameters are used to transform the data into desired state).

        2. Every transformer knows how it should be persisted and loaded (especially useful when
        working with Keras/Pytorch and Sklearn) in one pipeline.

    """

    def __init__(self):
        self.estimator = None

    def fit(self, *args, **kwargs):
        """Performs estimation of trainable parameters.

        All model estimations with sklearn, keras, pytorch models as well as some preprocessing
        techniques (normalization) estimate parameters based on data (training data).
        Those parameters are trained during fit execution and are persisted for the future.
        Only the estimation logic, nothing else.

        Args:
            args: positional arguments (can be anything)
            kwargs: keyword arguments (can be anything)

        Returns:
            BaseTransformer: self object
        """
        return self

    def transform(self, *args, **kwargs):
        """Performs transformation of data.

        All data transformation including prediction with deep learning/machine learning models
        can be performed here. No parameters should be estimated in this method nor stored as
        class attributes. Only the transformation logic, nothing else.

        Args:
            args: positional arguments (can be anything)
            kwargs: keyword arguments (can be anything)

        Returns:
            dict: outputs
        """
        raise NotImplementedError

    def fit_transform(self, *args, **kwargs):
        """Performs fit followed by transform.

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
        """Loads the trainable parameters of the transformer.

        Specific implementation of loading persisted model parameters should be implemented here.
        In case of transformers that do not learn any parameters one can leave this method as is.

        Args:
            filepath (str): filepath from which the transformer should be loaded
        Returns:
            BaseTransformer: self instance
        """
        return self

    def persist(self, filepath):
        """Saves the trainable parameters of the transformer

        Specific implementation of model parameter persistence should be implemented here.
        In case of transformers that do not learn any parameters one can leave this method as is.

        Args:
            filepath (str): filepath where the transformer parameters should be persisted
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
