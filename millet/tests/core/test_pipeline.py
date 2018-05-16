import pytest
from millet import MultiPipeline, SupervTransformer, UnsupervTransformer, DataLoader
from millet.core import MilletTypeException, MilletNameException


#
# Classes for testing
#


class SimpleSupervTransformer(SupervTransformer):
    def fit(self, input_dict: dict, superv_dict: {}) -> None:
        pass

    def transform(self, input_dict: dict) -> dict:
        output_dict = {'X': input_dict['X'], 'y': input_dict['y']}
        return output_dict


class SimpleUnsupervTransformer(UnsupervTransformer):
    def fit(self, input_dict: dict) -> None:
        pass

    def transform(self, input_dict: dict) -> dict:
        output_dict = {'X': input_dict['X']}
        return output_dict


class SimpleDataLoader(DataLoader):
    def load_data(self, input_info: dict) -> dict:
        output_dict = {'X': input_info['X']}
        return output_dict


#
# Tests
#


class TestPipeline(object):
    def test_add_dataloader(self):
        mpp = MultiPipeline()
        sdl = SimpleDataLoader()
        mpp.add_dataloader(sdl, 'sdl')
        assert mpp.get_step('sdl') == sdl

    # Check that an exception is raised when we try to add a dataloader but
    # pass a BaseTransformer instance
    @pytest.mark.xfail(raises=MilletTypeException)
    def test_add_dataloader_wrong_type(self):
        mpp = MultiPipeline()
        bad_dl = SimpleSupervTransformer()
        mpp.add_dataloader(bad_dl, 'bad_dl')

    @pytest.mark.xfail(raises=MilletNameException)
    def test_add_dataloader_repeat_name(self):
        mpp = MultiPipeline()
        sdl1 = SimpleDataLoader()
        sdl2 = SimpleDataLoader()
        mpp.add_dataloader(sdl1, 'sdl')
        mpp.add_dataloader(sdl2, 'sdl')

    def test_add_superv_transformer(self):
        mpp = MultiPipeline()
        sdl = SimpleDataLoader()
        mpp.add_dataloader(sdl, 'sdl')
        stfm = SimpleSupervTransformer()
        mpp.add_superv(stfm, 'stfm',
                       input_mapping={'X': ('sdl', 'X')},
                       superv_mapping={'y': ('sdl', 'y')}
                       )
        assert mpp.get_step('stfm') == stfm

    def test_add_unsuperv_transformer(self):
        mpp = MultiPipeline()
        sdl = SimpleDataLoader()
        mpp.add_dataloader(sdl, 'sdl')
        unstfm = SimpleUnsupervTransformer()
        mpp.add_unsuperv(unstfm, 'uns_tfm',
                         input_mapping={'X': ('sdl', 'X')},
                         )
        assert mpp.get_step('uns_tfm') == unstfm

    def test_add_superv_transformer_links_from_different_steps(self):
        pass

    # Check that an exception is raised when we try to add a transformer but
    # pass a BaseDataLoader instance
    @pytest.mark.xfail(raises=MilletTypeException)
    def test_add_superv_transformer_wrong_type(self):
        mpp = MultiPipeline()
        sdl = SimpleDataLoader()
        mpp.add_dataloader(sdl, 'sdl')
        bad_tfm = SimpleDataLoader()
        mpp.add_superv(bad_tfm, 'bad_tfm',
                       input_mapping={},
                       superv_mapping={}
                       )

    @pytest.mark.xfail(raises=MilletNameException)
    def test_add_superv_transformer_repeat_name(self):
        mpp = MultiPipeline()
        sdl = SimpleDataLoader()
        mpp.add_dataloader(sdl, 'sdl')
        stfm1 = SimpleSupervTransformer()
        stfm2 = SimpleSupervTransformer()
        mpp.add_superv(stfm1, 'stfm',
                       input_mapping={'X': ('sdl', 'X')},
                       superv_mapping={'y': ('sdl', 'y')}
                       )
        mpp.add_superv(stfm2, 'stfm',
                       input_mapping={'X': ('sdl', 'X')},
                       superv_mapping={'y': ('sdl', 'y')}
                       )

    def test_add_superv_transformer_creating_cycle(self):
        pass

    def test_string_conversion(self):
        pass

    def test_run(self):
        # Build a dummy graph and push some data through it...
        pass
