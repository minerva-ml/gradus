from ..core.base import DataLoader


class PassThroughDataLoader(DataLoader):
    """Most basic data loader step - prefer to use one with gatekeeping instead
    """

    def load_data(self, input_info):
        output_dict = input_info
        return output_dict


