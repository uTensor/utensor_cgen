import pytest


@pytest.fixture(name='config_user_values')
def config_user_values():
    from utensor_cgen.utils import Configuration

    return Configuration(
        defaults={
            'x': 1,
            'y': 2
        },
        user_config={
            'x': 2
        }
    )

@pytest.fixture(name='config_nested')
def config_nested():
    from utensor_cgen.utils import Configuration

    return Configuration(
        defaults={
            'dict1': {
                'inner': {
                    'x': 3,
                    'y': 4
                }
            }
        },
        user_config={
            'dict1': {
                'inner': {
                    'x': 2,
                }
            }
        }
    )
