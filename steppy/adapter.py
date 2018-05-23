from typing import Tuple, List, Dict, Any, NamedTuple

E = NamedTuple('E', [('input_name', str),
                     ('key', str)]
               )

AdaptingRecipe = Any
Results = Dict[str, Any]
AllInputs = Dict[str, Any]


class AdapterError(Exception):
    pass


class Adapter:
    def __init__(self, adapting_recipes: Dict[str, AdaptingRecipe]):
        self.adapting_recipes = adapting_recipes

    def adapt(self, all_inputs: AllInputs) -> Dict[str, Any]:
        adapted = {}
        for name, recipe in self.adapting_recipes.items():
            adapted[name] = self._construct(all_inputs, recipe)
        return adapted

    def _construct(self, all_inputs: AllInputs, recipe: AdaptingRecipe) -> Any:
        return {
            E: self._construct_element,
            tuple: self._construct_tuple,
            list: self._construct_list,
            dict: self._construct_dict,
        }.get(recipe.__class__, self._construct_constant)(all_inputs, recipe)

    def _construct_constant(self, _: AllInputs, constant) -> Any:
        return constant

    def _construct_element(self, all_inputs: AllInputs, element: E):
        input_name = element.input_name
        key = element.key
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

    def _construct_dict(self, all_inputs: AllInputs, dic: Dict[AdaptingRecipe, AdaptingRecipe]):
        return {self._construct(all_inputs, k): self._construct(all_inputs, v)
                for k, v in dic.items()}
