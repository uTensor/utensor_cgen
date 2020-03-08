def test_user_values(config_user_values):
    assert config_user_values['x'] == 2
    assert config_user_values['y'] == 2

def test_config_nested(config_nested):
    assert isinstance(config_nested['dict1'], type(config_nested))
    assert isinstance(config_nested['dict1']['inner'], type(config_nested))
    assert config_nested['dict1']['inner']['x'] == 2
