# TODO : rename this file


class TiedParameter:
    """Declare that a parameter is deterministically derived from another.

    Parameters:
        tied_to: Full dotted-path key of the source parameter
            (e.g. ``"spectrum.powerlaw_1.alpha"``).
        func: A callable ``f(source_value) -> derived_value``.

    Example::

        prior = {
            "spectrum.powerlaw_1.alpha": dist.Uniform(0, 5),
            "spectrum.powerlaw_2.alpha": TiedParameter(
                "spectrum.powerlaw_1.alpha", lambda x: 0.5 * x
            ),
        }
    """

    def __init__(self, tied_to: str, func):
        self.tied_to = tied_to
        self.func = func


class PerObs:
    """Mark a prior entry as having one independent value per observation.

    Two forms are supported:

    - ``PerObs(value)``: every observation independently draws from (or holds)
      the same distribution / fixed value (*homogeneous split*).
    - ``PerObs({obs_name: value, ...})``: each observation gets its own
      distribution / fixed value (*heterogeneous split*). Dict keys must
      cover every observation in the model.

    Any entry **not** wrapped in ``PerObs`` is shared across all observations
    (sampled once, broadcast).

    Example::

        from jaxspec.fit import PerObs
        import numpyro.distributions as dist

        prior = {
            # Shared — same value for every observation
            "spectrum.powerlaw_1.alpha": dist.Uniform(0, 5),
            # Split, homogeneous — independent draw per observation
            "spectrum.powerlaw_1.norm": PerObs(dist.LogUniform(1e-5, 1e-2)),
            # Split, heterogeneous — different prior per observation
            "spectrum.tbabs_1.nh": PerObs({
                "PN":   dist.Uniform(0, 1),
                "MOS1": dist.Uniform(0, 0.5),
            }),
        }
    """

    # TODO : Crash when using on a single obs
    def __init__(self, value):
        self.value = value

    @property
    def is_homogeneous(self) -> bool:
        """``True`` when all observations share the same prior/value."""
        return not isinstance(self.value, dict)
