from abc import ABC, abstractmethod


class BaseStep(ABC):
    pass


class BaseTransformer(BaseStep):
    @abstractmethod
    def fit(self, *input_dicts: [dict]) -> None:
        pass

    @abstractmethod
    def transform(self, input_dict: dict) -> dict:
        return {}


class SupervTransformer(BaseTransformer):

    @abstractmethod
    def fit(self, input_dict: dict, superv_dict: dict) -> None:
        pass

    @abstractmethod
    def transform(self, input_dict: dict) -> dict:
        return {}

    def fit_transform(self, input_dict: dict, superv_dict: dict) -> dict:
        self.fit(input_dict, superv_dict)
        return self.transform(input_dict)


class UnsupervTransformer(BaseTransformer):

    @abstractmethod
    def fit(self, input_dict: dict) -> None:
        pass

    @abstractmethod
    def transform(self, input_dict: dict) -> dict:
        return {}

    def fit_transform(self, input_dict: dict) -> dict:
        self.fit(input_dict)
        return self.transform(input_dict)


class DataLoader(BaseStep):
    """Base class for steps which load data
    """

    @abstractmethod
    def load_data(self, input_info: dict) -> dict:
        pass
