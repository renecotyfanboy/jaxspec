from collections.abc import Callable
from contextlib import contextmanager
from time import perf_counter


@contextmanager
def catchtime(desc="Task", print_time=True) -> Callable[[], float]:
    """
    Context manager to measure time taken by a task.

    Parameters
    ----------
        desc (str): Description of the task.
        print_time (bool): Whether to print the time taken by the task.

    Returns:
    -------
        Callable[[], float]: Function to get the time taken by the task.
    """

    t1 = t2 = perf_counter()
    yield lambda: t2 - t1
    t2 = perf_counter()
    if print_time:
        print(f"{desc}: {t2 - t1:.3f} seconds")
