from typing import Tuple, List, Dict, Any

AdaptingRecipe = Any
Results = Dict[str, Any]
AllInputs = Dict[str, Any]


class AdapterError(Exception):
    pass


class Adapter:
    """Translates outputs from parent steps to inputs to the current step

    Attributes:
        adapting_recipes: The recipes that the adapter was initialized with

    """
    def __init__(self, adapting_recipes: Dict[str, AdaptingRecipe]):
        """Adapter constructor

        Args:
            adapting_recipes: Recipes used to control the input translation.
                This should be a dict where the keys match the arguments
                expected by the transformer included in the step and the values
                can each be any of the following:

                1. ('input_name', 'key') will query the parent step
                    'input_name' for the output 'key'

                2. List of ('input_name', 'key') will query each parent step
                    with a name specified in 'input_name' for its respective
                    key and combine the results into a list

                3. Tuple of ('input_name', 'key') is the same as list of
                    ('input_name', 'key') except that a tuple is returned

                4. Dict like {k: ('input_name', 'key')} is the same as list of
                    ('input_name', 'key') except that a dict with keys k
                    is returned

                5. Anything else: the value itself will be used as the argument
                    to the transformer

        """
        self.adapting_recipes = adapting_recipes

    def adapt(self, all_inputs: AllInputs) -> Dict[str, Any]:
        """Adapt inputs for the transformer included in the step

        Args:
            all_inputs: Dict of outputs from parent steps. The keys should
            match the names of these steps and the values should be their
            respective outputs.

        Returns:
            adapted: Dict with the same keys as adapting_recipes and values
            constructed according to the respective recipes

        """
        adapted = {}
        for name, recipe in self.adapting_recipes.items():
            adapted[name] = self._construct(all_inputs, recipe)
        return adapted

    def _construct(self, all_inputs: AllInputs, recipe: AdaptingRecipe) -> Any:
        if isinstance(recipe, tuple):
            # Deduce if this is a basic recipe or a tuple of recipes
            if (len(recipe) == 2
                    and isinstance(recipe[0], str)
                    and isinstance(recipe[1], str)):
                res = self._construct_element(all_inputs, recipe)
            else:
                res = self._construct_tuple(all_inputs, recipe)
        elif isinstance(recipe, list):
            res = self._construct_list(all_inputs, recipe)
        elif isinstance(recipe, dict):
            res = self._construct_dict(all_inputs, recipe)
        else:
            res = self._construct_constant(all_inputs, recipe)

        return res

    def _construct_constant(self, _: AllInputs, constant) -> Any:
        return constant

    def _construct_element(self, all_inputs: AllInputs, element: Tuple[str, str]):
        input_name = element[0]
        key = element[1]
        try:
            input_results = all_inputs[input_name]
            try:
                return input_results[key]
            except KeyError:
                msg = "Input '{}' didn't have '{}' in its result.".format(input_name, key)
                raise AdapterError(msg)
        except KeyError:
            msg = "No such input: '{}'".format(input_name)
            raise AdapterError(msg)

    def _construct_list(self, all_inputs: AllInputs, lst: List[AdaptingRecipe]):
        return [self._construct(all_inputs, recipe) for recipe in lst]

    def _construct_tuple(self, all_inputs: AllInputs, tup: Tuple):
        return tuple(self._construct(all_inputs, recipe) for recipe in tup)

    def _construct_dict(self, all_inputs: AllInputs, dic: Dict[Any, AdaptingRecipe]):
        return {self._construct(all_inputs, k): self._construct(all_inputs, v)
                for k, v in dic.items()}
