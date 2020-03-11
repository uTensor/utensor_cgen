import time


def test_cli_load_speed():
    from utensor_cgen.cli import cli

    durations = []
    for _ in range(1000):
        start_time = time.time()
        cli.main(args=['-h'], standalone_mode=False, )
        end_time = time.time()
        durations.append(end_time - start_time)
    mean_duration = sum(durations) / len(durations)
    assert mean_duration <= 0.005, 'cli is too slow: {:0.5f}'.format(mean_duration)
