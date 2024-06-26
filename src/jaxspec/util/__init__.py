from collections.abc import Callable
from contextlib import contextmanager
from time import perf_counter

import haiku as hk

from jax.random import PRNGKey, split


@contextmanager
def catchtime(desc="Task", print_time=True) -> Callable[[], float]:
    """
    Context manager to measure time taken by a task.

    Parameters
    ----------
        desc (str): Description of the task.
        print_time (bool): Whether to print the time taken by the task.

    Returns
    -------
        Callable[[], float]: Function to get the time taken by the task.
    """

    t1 = t2 = perf_counter()
    yield lambda: t2 - t1
    t2 = perf_counter()
    if print_time:
        print(f"{desc}: {t2 - t1:.3f} seconds")


def sample_prior(dict_of_prior, key=PRNGKey(0), flat_parameters=False):
    """
    Sample the prior distribution from a dict of distributions
    """

    parameters = dict(hk.data_structures.to_haiku_dict(dict_of_prior))
    parameters_flat = {}

    for m, n, distribution in hk.data_structures.traverse(dict_of_prior):
        key, subkey = split(key)
        parameters[m][n] = distribution.sample(subkey)
        parameters_flat[m + "_" + n] = distribution.sample(subkey)

    return parameters if not flat_parameters else parameters_flat
