from utensor_cgen.utils import NamescopedKWArgsParser


def test_kwarg_parser():
    op_attr = {
        'global': 10,
        'var1': 1,
        'private__var1': 2,
    }
    parser = NamescopedKWArgsParser('private', op_attr)
    assert parser.get('no_such_thing') is None
    assert parser.get('global') == 10
    assert parser.get('var1') == 2
    try:
        parser['no_such_thing']
    except KeyError:
        pass
