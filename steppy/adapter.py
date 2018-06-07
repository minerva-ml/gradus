from typing import Tuple, List, Dict, Any, NamedTuple

E = NamedTuple('E', [('input_name', str),
                     ('key', str)]
               )

AdaptingRecipe = Any
DataPacket = Dict[str, Any]
AllOutputs = Dict[str, DataPacket]


class AdapterError(Exception):
    pass


class Adapter:
    """Translates outputs from parent steps to inputs to the current step.

    Attributes:
        adapting_recipes: The recipes that the adapter was initialized with.

    Example:
        Normally Adapter is used with a Step. In the following example
        `RandomForestTransformer` follows sklearn convention of calling arguments `X` and `y`,
        however names passed to the Step are different. We use Adapter to map recieved names
        to the expected names.

        .. code-block:: python

            from sklearn.datasets import load_iris
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import log_loss
            from steppy.base import BaseTransformer, Step
            from steppy.adapter import Adapter, E

            iris = load_iris()

            pipeline_input = {
                'train_data': {
                    'target': iris.target,
                    'data': iris.data
                }
            }

            class RandomForestTransformer(BaseTransformer):
                def __init__(self, random_state=None):
                    self.estimator = RandomForestClassifier(random_state=random_state)

                def fit(self, X, y):
                    self.estimator.fit(X, y)
                    return self

                def transform(self, X, **kwargs):
                    y_proba  = self.estimator.predict_proba(X)
                    return {'y_proba': y_proba}

            random_forest = Step(
                name="random_forest",
                transformer=RandomForestTransformer(),
                input_data=['train_data'],
                adapter=Adapter({
                    'X': E('train_data', 'data'),
                    'y': E('train_data', 'target')
                }),
                experiment_directory='./working_dir'
            )

            result = random_forest.fit_transform(pipeline_input)
            print(log_loss(y_true=iris.target, y_pred=result['y_proba']))
    """

    def __init__(self, adapting_recipes: Dict[str, AdaptingRecipe]):
        """Adapter constructor.

        Note:
            You have to import the extractor 'E' from this module to construct
            adapters.

        Args:
            adapting_recipes: Recipes used to control the input translation.
                An adapting recipe may be any Python data structure. If this structure
                contains placeholders denoted by `E`, then values extracted from parent
                steps' outputs will be substituted in their place.
                `adapting_recipes` is a dict where the keys match the arguments
                expected by the transformer. The values in this dictionary may be for example
                one of the following:

                1. `E('input_name', 'key')` will query the parent step
                    'input_name' for the output 'key'

                2. List of `E('input_name', 'key')` will apply the extractors
                    to the parent steps and combine the results into a list

                3. Tuple of `E('input_name', 'key')` will apply the extractors
                    to the parent steps and combine the results into a tuple

                4. Dict like `{k: E('input_name', 'key')}` will apply the
                    extractors to the parent steps and combine the results
                    into a dict with the same keys

                5. Anything else: the value itself will be used as the argument
                    to the transformer
        """
        self.adapting_recipes = adapting_recipes

    def adapt(self, all_ouputs: AllOutputs) -> DataPacket:
        """Adapt inputs for the transformer included in the step.

        Args:
            all_ouputs: Dict of outputs from parent steps. The keys should
                match the names of these steps and the values should be their
                respective outputs.

        Returns:
            Dictionary with the same keys as `adapting_recipes` and values
            constructed according to the respective recipes.

        """
        adapted = {}
        for name, recipe in self.adapting_recipes.items():
            adapted[name] = self._construct(all_ouputs, recipe)
        return adapted

    def _construct(self, all_ouputs: AllOutputs, recipe: AdaptingRecipe) -> Any:
        return {
            E: self._construct_element,
            tuple: self._construct_tuple,
            list: self._construct_list,
            dict: self._construct_dict,
        }.get(recipe.__class__, self._construct_constant)(all_ouputs, recipe)

    def _construct_constant(self, _: AllOutputs, constant) -> Any:
        return constant

    def _construct_element(self, all_ouputs: AllOutputs, element: E):
        input_name = element.input_name
        key = element.key
        try:
            input_results = all_ouputs[input_name]
            try:
                return input_results[key]
            except KeyError:
                msg = "Input '{}' didn't have '{}' in its result.".format(input_name, key)
                raise AdapterError(msg)
        except KeyError:
            msg = "No such input: '{}'".format(input_name)
            raise AdapterError(msg)

    def _construct_list(self, all_ouputs: AllOutputs, lst: List[AdaptingRecipe]):
        return [self._construct(all_ouputs, recipe) for recipe in lst]

    def _construct_tuple(self, all_ouputs: AllOutputs, tup: Tuple):
        return tuple(self._construct(all_ouputs, recipe) for recipe in tup)

    def _construct_dict(self, all_ouputs: AllOutputs, dic: Dict[AdaptingRecipe, AdaptingRecipe]):
        return {self._construct(all_ouputs, k): self._construct(all_ouputs, v)
                for k, v in dic.items()}
