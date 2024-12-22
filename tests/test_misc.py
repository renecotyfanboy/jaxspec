from jaxspec.util.misc import catchtime


def test_func():
    return 1


def test_catchtime_measures_time_correctly():
    with catchtime(desc="Test Task", print_time=False) as get_time:
        result = test_func()
    assert result == 1
    assert get_time() > 0


def test_catchtime_prints_time_when_print_time_is_true(capfd):
    with catchtime(desc="Test Task", print_time=True) as get_time:
        result = test_func()
    out, err = capfd.readouterr()
    assert "Test Task" in out
    assert "seconds" in out


def test_catchtime_does_not_print_time_when_print_time_is_false(capfd):
    with catchtime(desc="Test Task", print_time=False) as get_time:
        result = test_func()
    out, err = capfd.readouterr()
    assert out == ""
