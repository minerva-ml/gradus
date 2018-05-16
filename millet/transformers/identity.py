from ..core import BaseTransformer


class IdentityTransformer(BaseTransformer):
    def fit(self, input_dict: dict) -> None:
        pass

    def transform(self, input_dict: dict) -> dict:
        return input_dict
