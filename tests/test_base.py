import pytest

from steps.base import make_transformer


@pytest.mark.parametrize("mode", [0, 1])
def test_make_transformer(mode):
    def fun(x, y, mode=0):
        return x + y if mode == 0 else x - y
    tr = make_transformer(fun)

    tr.fit()
    res = tr.transform(7, 3, mode=mode)
    assert res == (10 if mode == 0 else 4)
