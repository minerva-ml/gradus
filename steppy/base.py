import os
import pprint
from collections import defaultdict

from sklearn.externals import joblib

from steppy.adapter import Adapter, AdapterError
from steppy.utils import display_upstream_structure, persist_as_png, get_logger, initialize_logger

initialize_logger()
logger = get_logger()

DEFAULT_TRAINING_SETUP = {
    'is_fittable': True,
    'force_fitting': True,
    'persist_output': False,
    'cache_output': False,
    'load_persisted_output': False
}


class Step:
    """Step is a building block of steppy pipelines.

    It is an execution wrapper over the transformer (see :class:`~steppy.base.BaseTransformer`),
    which realizes single operation on data. With Step you can:

    1. design multiple input/output data flows and connections between Steps.
    2. handle persistence and caching of transformer and intermediate results.

    Step executes `fit_transform` method inspired by the sklearn on every step recursively
    starting from the very last Step and making its way forward through the `input_steps`.
    One can easily debug the data flow by plotting the pipeline graph
    (see: :func:`~steppy.utils.persist_as_png`) or return step in a jupyter notebook cell.

    Attributes:
        transformer (obj): object that inherits from BaseTransformer or Step instance.
            When Step instance is passed, transformer from that Step will be copied and used to
            perform transformations. It is useful when both train and valid data are passed in
            one pipeline (common situation in deep learning).

        name (str): Step name.
            Each step in a pipeline must have a unique name. It is name of the persisted
            transformer and output of this Step.
            Default is transformer's class name.

        experiment_directory (str): path to the directory where all execution artifacts will be
            stored.
            Default is ``~/.steppy``.
            The following directories will be created under ``~/.steppy``, if they were not created by
            preceding Steps:

            * transformers: transformer objects are persisted in this folder
            * output:      step output dictionaries are persisted in this folder
              (if ``persist_output=True``)

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
            Current Step will combine output from `input_steps` and `input_data` using `adapter`.
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

        cache_output (bool): If True, Step output dictionary will be cached under
            ``self.output``, when transform method of the Step transformer
            is completed. If the same Step is used multiple times, transform method is invoked
            only once. Further invokes simply use cached output.
            Default ``False``: do not cache output

            Warning:
                One should always run `step.clean_cache_upstream()` before executing
                `step.fit_transform(data)` or `step.transform(data)`
                When working with large datasets, cache might be very large.

        persist_output (bool): If True, persist Step output to disk under the
            ``<experiment_directory>/output/<name>`` directory.
            Default ``False``: do not persist any files to disk.
            If True then Step output dictionary will be persisted to the
            ``<experiment_directory>/output/<name>`` directory, after transform method of the Step
            transformer is completed. Step persists to disk the output after every run of the
            transformer's transform method. It means that Step overrides files. See also
            `load_persisted_output` parameter.

            Warning:
                When working with large datasets, cache might be very large.

        load_persisted_output (bool): If True, Step output dictionary already persisted to the
            ``<experiment_directory>/output/<name>`` will be loaded when Step is called.
            Default ``False``: do not load persisted output.
            Useful when debugging and working with ensemble models or time consuming feature
            extraction. One can easily persist already computed pieces of the pipeline and save
            time by loading them instead of calculating.

            Warning:
                Re-running the same step on new data with `load_persisted_output` set ``True``
                may lead to errors when output from old data are loaded while user would expect
                the pipeline to use new data instead.

        force_fitting (bool): If True, Step transformer will be fitted (via `fit_transform`)
            even if ``<experiment_directory>/transformers/<step_name>`` exists.
            Default ``True``: fit transformer each time `fit_transform()` is called.
            Helpful when one wants to use ``persist_output=True`` and load ``persist_output=True``
            on a previous Step and fit current Step multiple times. This is a typical scenario
            for tuning hyperparameters for an ensemble model trained on the output from first
            level models or a model build on features that are time consuming to compute.
    """

    def __init__(self,
                 transformer,
                 name=None,
                 experiment_directory=None,
                 output_directory=None,
                 input_data=None,
                 input_steps=None,
                 adapter=None,

                 is_fittable=True,
                 force_fitting=True,

                 persist_output=False,
                 cache_output=False,
                 load_persisted_output=False):

        self.name = self._format_step_name(name, transformer)

        if experiment_directory is not None:
            assert isinstance(experiment_directory, str),\
                'Step {} error, experiment_directory must ' \
                'be str, got {} instead.'.format(self.name, type(experiment_directory))
        else:
            experiment_directory = os.path.join(os.path.expanduser("~"), '.steppy')
            logger.info('Using default experiment directory: {}'.format(experiment_directory))

        if output_directory is not None:
            assert isinstance(output_directory, str),\
                'Step {}, output_directory must be str, got {} instead'.format(self.name, type(output_directory))

        if input_data is not None:
            assert isinstance(input_data, list), 'Step {} error, input_data must be list, ' \
                                                 'got {} instead.'.format(self.name, type(input_data))
        if input_steps is not None:
            assert isinstance(input_steps, list), 'Step {} error, input_steps must be list, ' \
                                                  'got {} instead.'.format(self.name, type(input_steps))
        if adapter is not None:
            assert isinstance(adapter, Adapter), 'Step {} error, adapter must be an instance ' \
                                                 'of {}'.format(self.name, str(Adapter))

        assert isinstance(cache_output, bool), 'Step {} error, cache_output must be bool, ' \
                                               'got {} instead.'.format(self.name, type(cache_output))
        assert isinstance(persist_output, bool), 'Step {} error, persist_output must be bool, ' \
                                                 'got {} instead.'.format(self.name, type(persist_output))
        assert isinstance(load_persisted_output, bool),\
            'Step {} error, load_persisted_output ' \
            'must be bool, got {} instead.'.format(self.name, type(load_persisted_output))
        assert isinstance(force_fitting, bool), 'Step {} error, force_fitting must be bool, ' \
                                                'got {} instead.'.format(self.name, type(force_fitting))

        logger.info('Initializing Step {}'.format(self.name))

        self.transformer = transformer
        self.output_directory = output_directory
        self.input_steps = input_steps or []
        self.input_data = input_data or []
        self.adapter = adapter
        self.is_fittable = is_fittable
        self.cache_output = cache_output
        self.persist_output = persist_output
        self.load_persisted_output = load_persisted_output
        self.force_fitting = force_fitting

        self.output = None
        self.experiment_directory = os.path.join(experiment_directory)
        self._prepare_experiment_directories()
        self._mode = 'train'

        self._validate_upstream_names()
        logger.info('Step {} initialized'.format(self.name))

    @property
    def experiment_directory_transformers_step(self):
        directory = os.path.join(self.experiment_directory, 'transformers')
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, self.name)

    @property
    def experiment_directory_output_step(self):
        directory = os.path.join(self.experiment_directory, 'output')
        if self.output_directory is not None:
            os.makedirs(os.path.join(directory, self.output_directory), exist_ok=True)
            return os.path.join(directory, self.output_directory, self.name)

        if self._mode == 'train':
            os.makedirs(os.path.join(directory, 'train'), exist_ok=True)
            return os.path.join(directory, 'train', self.name)

        if self._mode == 'inference':
            os.makedirs(os.path.join(directory, 'inference'), exist_ok=True)
            return os.path.join(directory, 'inference', self.name)

    @property
    def upstream_structure(self):
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
    def all_upstream_steps(self):
        """Build dictionary with all Step instances that are upstream to `self`.

        Returns:
            all_upstream_steps (dict): dictionary where keys are Step names (str) and values are Step
            instances (obj)
        """
        all_steps_ = {}
        all_steps_ = self._get_steps(all_steps_)
        return all_steps_

    @property
    def transformer_is_persisted(self):
        """(bool): True if transformer exists under the directory
        ``<experiment_directory>/transformers/<step_name>``
        """
        return os.path.exists(self.experiment_directory_transformers_step)

    @property
    def output_is_cached(self):
        """(bool): True if step output exists under the ``self.output``.
            See `cache_output`.
        """
        if self.output is not None:
            return True
        else:
            return False

    @property
    def output_is_persisted(self):
        """(bool): True if step output exists under the ``<experiment_directory>/output/<mode>/<name>``.
            See :attr:`~steppy.base.Step.persist_output`.
        """
        return os.path.exists(self.experiment_directory_output_step)

    def fit_transform(self, data):
        """Fit the model and transform data or load already processed data.

        Loads cached or persisted output or adapts data for the current transformer and
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
            dict: Step output from the ``self.transformer.fit_transform`` method
        """
        logger.info('Step {}, working in "{}" mode'.format(self.name, self._mode))
        if self._mode == 'inference':
            ValueError('Step {}, you are in "{}" mode, where you cannot run "fit".'
                       'Please change mode to "train" to enable fitting.'
                       'Use: "step.set_mode_train()" then "step.fit_transform()"'.format(self.name, self._mode))

        if self.output_is_cached and not self.force_fitting:
            logger.info('Step {} using cached output'.format(self.name))
            step_output_data = self.output
        elif self.output_is_persisted and self.load_persisted_output and not self.force_fitting:
            logger.info('Step {} loading persisted output from {}'.format(self.name,
                                                                          self.experiment_directory_output_step))
            step_output_data = self._load_output(self.experiment_directory_output_step)
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
            step_output_data = self._fit_transform_operation(step_inputs)
        logger.info('Step {}, fit and transform completed'.format(self.name))
        return step_output_data

    def transform(self, data):
        """Transforms data or loads already processed data.

        Loads cached persisted output or adapts data for the current transformer and executes
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
            dict: step output from the transformer.transform method
        """
        logger.info('Step {}, working in "{}" mode'.format(self.name, self._mode))
        if self.output_is_cached:
            logger.info('Step {} using cached output'.format(self.name))
            step_output_data = self.output
        elif self.output_is_persisted and self.load_persisted_output:
            logger.info('Step {} loading persisted output from {}'.format(self.name,
                                                                          self.experiment_directory_output_step))
            step_output_data = self._load_output(self.experiment_directory_output_step)
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
            step_output_data = self._transform_operation(step_inputs)
        logger.info('Step {}, transform completed'.format(self.name))
        return step_output_data

    def set_mode_train(self):
        """Applies 'train' mode to all upstream Steps including this Step
        and cleans cache for all upstream Steps including this Step.
        """
        self._set_mode('train')
        return self

    def set_mode_inference(self):
        """Applies 'inference' mode to all upstream Steps including this Step
        and cleans cache for all upstream Steps including this Step.
        """
        self._set_mode('inference')
        return self

    def reset(self):
        """Reset all upstream Steps to the default training parameters and
        cleans cache for all upstream Steps including this Step.
        Defaults are:
            'mode': 'train',
            'is_fittable': True,
            'force_fitting': True,
            'persist_output': False,
            'cache_output': False,
            'load_persisted_output': False
        """
        self.clean_cache_upstream()
        self.set_mode_train()
        for step_obj in self.all_upstream_steps.values():
            step_obj.is_fittable = DEFAULT_TRAINING_SETUP['is_fittable']
            step_obj.force_fitting = DEFAULT_TRAINING_SETUP['force_fitting']
            step_obj.persist_output = DEFAULT_TRAINING_SETUP['persist_output']
            step_obj.cache_output = DEFAULT_TRAINING_SETUP['cache_output']
            step_obj.load_persisted_output = DEFAULT_TRAINING_SETUP['load_persisted_output']
        logger.info('Step {}, reset all upstream Steps to default training parameters, '
                    'including this Step'.format(self.name))
        return self

    def set_parameters_upstream(self, parameters):
        """Set parameters to all upstream Steps including this Step.
        Parameters is dict() where key is Step attribute, and value is new value to set.
        """
        assert isinstance(parameters, dict), 'parameters must be dict, got {} instead'.format(type(parameters))
        for step_obj in self.all_upstream_steps.values():
            for key in step_obj.__dict__.keys():
                if key in list(parameters.keys()):
                    step_obj.__dict__[key] = parameters[key]
                    if key == 'experiment_directory':
                        step_obj._prepare_experiment_directories()
        logger.info('set new values to all upstream Steps including this Step.')
        return self

    def clean_cache_step(self):
        """Clean cache for current step.
        """
        logger.info('Step {}, cleaning cache'.format(self.name))
        self.output = None
        return self

    def clean_cache_upstream(self):
        """Clean cache for all steps that are upstream to `self`.
        """
        logger.info('Cleaning cache for the entire upstream pipeline')
        for step in self.all_upstream_steps.values():
            logger.info('Step {}, cleaning cache'.format(step.name))
            step.output = None
        return self

    def get_step_by_name(self, name):
        """Extracts step by name from the pipeline.

        Extracted Step is a fully functional pipeline as well.
        All upstream Steps are already defined.

        Args:
            name (str): name of the step to be fetched
        Returns:
            Step (obj): extracted step
        """
        self._validate_step_name(name)
        name = str(name)
        try:
            return self.all_upstream_steps[name]
        except KeyError as e:
            msg = 'No Step with name "{}" found. ' \
                  'You have following Steps: {}'.format(name, list(self.all_upstream_steps.keys()))
            raise StepError(msg) from e

    def persist_upstream_structure(self):
        """Persist json file with the upstream steps structure, that is step names and their connections."""
        persist_dir = os.path.join(self.experiment_directory, '{}_upstream_structure.json'.format(self.name))
        logger.info('Step {}, saving upstream pipeline structure to {}'.format(self.name, persist_dir))
        joblib.dump(self.upstream_structure, persist_dir)

    def persist_upstream_diagram(self, filepath):
        """Creates upstream steps diagram and persists it to disk as png file.

        Pydot graph is created and persisted to disk as png file under the filepath directory.

        Args:
            filepath (str): filepath to which the png with steps visualization should
                be persisted
        """
        assert isinstance(filepath, str),\
            'Step {} error, filepath must be str. Got {} instead'.format(self.name, type(filepath))
        persist_as_png(self.upstream_structure, filepath)

    def _fit_transform_operation(self, step_inputs):
        if self.is_fittable:
            if self.transformer_is_persisted and not self.force_fitting:
                logger.info('Step {}, loading transformer from the {}'
                            .format(self.name, self.experiment_directory_transformers_step))
                self.transformer.load(self.experiment_directory_transformers_step)
                logger.info('Step {}, transforming...'.format(self.name))

                try:
                    step_output_data = self.transformer.transform(**step_inputs)
                except Exception as e:
                    msg = 'Step {}, Transformer "{}" error ' \
                          'during "transform()" operation.'.format(self.name, self.transformer.__class__.__name__)
                    raise StepError(msg) from e

                logger.info('Step {}, transforming completed'.format(self.name))
            else:
                logger.info('Step {}, fitting and transforming...'.format(self.name))

                try:
                    step_output_data = self.transformer.fit_transform(**step_inputs)
                except Exception as e:
                    msg = 'Step {}, Transformer "{}" error ' \
                          'during "fit_transform()" operation.'.format(self.name, self.transformer.__class__.__name__)
                    raise StepError(msg) from e

                logger.info('Step {}, fitting and transforming completed'.format(self.name))
                logger.info('Step {}, persisting transformer to the {}'
                            .format(self.name, self.experiment_directory_transformers_step))
                self.transformer.persist(self.experiment_directory_transformers_step)
        else:
            logger.info('Step {}, is not fittable, transforming...'.format(self.name))

            try:
                step_output_data = self.transformer.transform(**step_inputs)
            except Exception as e:
                msg = 'Step {}, Transformer "{}" error ' \
                      'during "transform()" operation.'.format(self.name, self.transformer.__class__.__name__)
                raise StepError(msg) from e

            logger.info('Step {}, transforming completed'.format(self.name))
        if self.cache_output:
            logger.info('Step {}, caching output'.format(self.name))
            self.output = step_output_data
        if self.persist_output:
            logger.info('Step {}, persisting output to the {}'
                        .format(self.name, self.experiment_directory_output_step))
            self._persist_output(step_output_data, self.experiment_directory_output_step)
        return step_output_data

    def _transform_operation(self, step_inputs):
        if self.is_fittable:
            if self.transformer_is_persisted:
                logger.info('Step {}, loading transformer from the {}'
                            .format(self.name, self.experiment_directory_transformers_step))
                self.transformer.load(self.experiment_directory_transformers_step)
                logger.info('Step {}, transforming...'.format(self.name))

                try:
                    step_output_data = self.transformer.transform(**step_inputs)
                except Exception as e:
                    msg = 'Step {}, Transformer "{}" error ' \
                          'during "transform()" operation.'.format(self.name, self.transformer.__class__.__name__)
                    raise StepError(msg) from e

                logger.info('Step {}, transforming completed'.format(self.name))
            else:
                raise ValueError('No transformer persisted with name: {}'
                                 'Make sure that you have this transformer under the directory: {}'
                                 .format(self.name, self.experiment_directory_transformers_step))
        else:
            logger.info('Step {}, transforming...'.format(self.name))

            try:
                step_output_data = self.transformer.transform(**step_inputs)
            except Exception as e:
                msg = 'Step {}, Transformer "{}" error ' \
                      'during "transform()" operation.'.format(self.name, self.transformer.__class__.__name__)
                raise StepError(msg) from e

            logger.info('Step {}, transforming completed'.format(self.name))
        if self.cache_output:
            logger.info('Step {}, caching output'.format(self.name))
            self.output = step_output_data
        if self.persist_output:
            logger.info('Step {}, persisting output to the {}'
                        .format(self.name, self.experiment_directory_output_step))
            self._persist_output(step_output_data, self.experiment_directory_output_step)
        return step_output_data

    def _load_output(self, filepath):
        logger.info('Step {}, loading output from {}'.format(self.name, filepath))
        return joblib.load(filepath)

    def _persist_output(self, output_data, filepath):
        joblib.dump(output_data, filepath)

    def _adapt(self, step_inputs):
        logger.info('Step {}, adapting inputs'.format(self.name))
        try:
            return self.adapter.adapt(step_inputs)
        except AdapterError as e:
            msg = "Error while adapting step '{}'. Check Step inputs".format(self.name)
            raise StepError(msg) from e

    def _unpack(self, step_inputs):
        logger.info('Step {}, unpacking inputs'.format(self.name))
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
            msg = "Could not unpack inputs. Following keys are present in multiple input steps:\n " \
                  "\n".join(["  '{}' present in steps {}".format(key, step_names)
                             for key, step_names in repeated_keys])
            raise StepError(msg)

    def _prepare_experiment_directories(self):
        if not os.path.exists(os.path.join(self.experiment_directory, 'transformers')):
            logger.info('initializing experiment directories under {}'.format(self.experiment_directory))
            for dir_name in ['transformers', 'output']:
                os.makedirs(os.path.join(self.experiment_directory, dir_name), exist_ok=True)

    def _get_steps(self, all_steps):
        self._check_name_uniqueness(all_steps=all_steps)
        for input_step in self.input_steps:
            all_steps = input_step._get_steps(all_steps)
        all_steps[self.name] = self
        return all_steps

    def _format_step_name(self, name, transformer):
        self._validate_step_name(name=name)
        if name is not None:
            name_ = str(name)
        else:
            name_ = transformer.__class__.__name__
        return name_

    def _validate_step_name(self, name):
        if name is not None:
            assert isinstance(name, str) or isinstance(name, float) or isinstance(name, int),\
                'Step name must be str, float or int. Got {} instead.'.format(type(name))

    def _check_name_uniqueness(self, all_steps):
        if self.name in all_steps.keys():
            logger.info('STEPPY WARNING: Step with name "{}", already exist. '
                        'Make sure that all Steps have unique name.'.format(self.name))

    def _validate_upstream_names(self):
        try:
            _ = self.all_upstream_steps.keys()
        except ValueError as e:
            msg = 'Incorrect Step names'
            raise StepError(msg) from e

    def _build_structure_dict(self, structure_dict):
        for input_step in self.input_steps:
            structure_dict = input_step._build_structure_dict(structure_dict)
            structure_dict['edges'].add((input_step.name, self.name))
        structure_dict['nodes'].add(self.name)
        for input_data in self.input_data:
            structure_dict['nodes'].add(input_data)
            structure_dict['edges'].add((input_data, self.name))
        return structure_dict

    def _set_mode(self, mode):
        self.clean_cache_upstream()
        for name, step_obj in self.all_upstream_steps.items():
            step_obj._mode = mode
        logger.info('Step {}, applied "{}" mode to all upstream Steps, including this Step'.format(self.name, mode))

    def _repr_html_(self):
        return display_upstream_structure(self.upstream_structure)

    def __str__(self):
        return pprint.pformat(self.upstream_structure)


class BaseTransformer:
    """Abstraction on ``fit`` and ``transform`` execution.

    Base transformer is an abstraction strongly inspired by the ``sklearn.Transformer`` and
    ``sklearn.Estimator``. Two main concepts are:

        1. Every action that can be performed on data (transformation, model training) can be
        performed in two steps: fitting (where trainable parameters are estimated) and transforming
        (where previously estimated parameters are used to transform the data into desired state).

        2. Every transformer knows how it should be persisted and loaded (especially useful when
        working with Keras/Pytorch or scikit-learn) in one pipeline.
    """

    def __init__(self):
        self.estimator = None

    def fit(self, *args, **kwargs):
        """Performs estimation of trainable parameters.

        All model estimations with scikit-learn, keras, pytorch models as well as some preprocessing
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
            dict: output
        """
        raise NotImplementedError

    def fit_transform(self, *args, **kwargs):
        """Performs fit followed by transform.

        This method simply combines fit and transform.

        Args:
            args: positional arguments (can be anything)
            kwargs: keyword arguments (can be anything)

        Returns:
            dict: output
        """
        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)

    def load(self, filepath):
        """Loads the trainable parameters of the transformer.

        Specific implementation of loading persisted model parameters should be implemented here.
        In case of transformer that do not learn any parameters one can leave this method as is.

        Args:
            filepath (str): filepath from which the transformer should be loaded
        Returns:
            BaseTransformer: self instance
        """
        _ = filepath
        return self

    def persist(self, filepath):
        """Saves the trainable parameters of the transformer

        Specific implementation of model parameter persistence should be implemented here.
        In case of transformer that do not learn any parameters one can leave this method as is.

        Args:
            filepath (str): filepath where the transformer parameters should be persisted
        """
        joblib.dump('hello-steppy', filepath)


class StepError(Exception):
    pass


def make_transformer(func):
    class StaticTransformer(BaseTransformer):
        def fit(self):
            logger.info('StaticTransformer "{}" is not fittable.'
                        'By running "fit_transform()", you simply "transform()".'.format(self.__class__.__name__))
            return self

        def transform(self, *args, **kwargs):
            return func(*args, **kwargs)

        def persist(self, filepath):
            logger.info('StaticTransformer "{}" is not persistable.'.format(self.__class__.__name__))

    _transformer = StaticTransformer()
    _transformer.__class__.__name__ = func.__name__
    return _transformer


class IdentityOperation(BaseTransformer):
    """Transformer that performs identity operation, f(x)=x."""

    def transform(self, **kwargs):
        return kwargs

    def persist(self, filepath):
        logger.info('"IdentityOperation" is not persistable.')
        pass
