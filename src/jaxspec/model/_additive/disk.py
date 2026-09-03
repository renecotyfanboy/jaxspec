"""Multicolor disk blackbody additive models."""

from __future__ import annotations

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np

from jax.scipy.special import logsumexp

from ..abc import AdditiveComponent

DEFAULT_FLUX_BAND = (0.5, 10.0)
"""Band used to normalize the disk models when none is given, in keV."""

STANDARD_RADIAL_EXPONENT = 0.75
"""Radial temperature exponent of the standard multicolor disk model."""


def _usable_log_max(dtype) -> float:
    """Largest exponent leaving headroom for the response fold downstream."""
    return float(np.log(np.float64(np.finfo(dtype).max)) - 30.0)


def _exp_normalized(log_ratio, norm):
    """Return ``norm * exp(log_ratio)``, capped so the product stays finite.

    The cap applies to the scaled value, so a small ``norm`` can bring a large ratio
    into range, and the result stays an exact multiple of ``norm``, including zero.
    """
    norm = jnp.asarray(norm)
    abs_norm = jnp.abs(norm)
    log_norm = jnp.log(jnp.where(abs_norm > 0.0, abs_norm, 1.0))
    log_max = _usable_log_max(jnp.asarray(log_ratio).dtype)
    return norm * jnp.exp(jnp.minimum(log_ratio + log_norm, log_max) - log_norm)


def _disk_quadrature(n: int = 32, half_width: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Return tanh-sinh coordinates and integration weights.

    Coordinates store ``-log(y)`` so extreme nodes do not round to zero or one.
    """
    t = np.linspace(-half_width, half_width, n)
    z = np.pi / 2 * np.sinh(t)
    c = np.logaddexp(0.0, -2.0 * z)
    w = np.pi / 2 * np.cosh(t) / np.cosh(z) ** 2 / 2.0 * (t[1] - t[0])
    w[0] *= 0.5
    w[-1] *= 0.5
    return c, w


_DISK_QUAD_C, _DISK_QUAD_W = _disk_quadrature()
_DISK_GL_X, _DISK_GL_W = np.polynomial.legendre.leggauss(64)


def _disk_quad_log_shape(energy, tin, p):
    r"""Reduced log disk shape by tanh-sinh quadrature, with its positive-energy mask.

    $$F(E) = 2.78 \times 10^{-3} (0.75/p) T_{\rm in}^2 t^{2-s}e^{-t}
    \int_0^1 \frac{(t-\ln y)^{s-1}}{1-ye^{-t}}\,\mathrm{d}y,
    \quad t=E/T_{\rm in},\ s=2/p.$$

    The prefactor in front of $t^{2-s}$ is left out: it cancels against the band
    integral. Non-positive energies evaluate on a finite branch, to be masked.
    """
    s = 2.0 / jnp.asarray(p)
    t = energy / tin
    t_safe = jnp.where(t > 0.0, t, 1.0)
    u = t_safe[..., None] + _DISK_QUAD_C
    integral = jnp.sum(_DISK_QUAD_W * u ** (s[..., None] - 1.0) / (-jnp.expm1(-u)), axis=-1)
    return (2.0 - s) * jnp.log(t_safe) - t_safe + jnp.log(integral), t > 0.0


def _diskbb_psi_table(
    n_tab: int = 2048, t_lo: float = 1e-8, t_hi: float = 745.0
) -> tuple[float, float, np.ndarray, np.ndarray]:
    r"""Tabulate $\psi(\tau)=\ln[G(t)e^t t^{-(s-1)}]$ at $p=0.75$, uniform in $\tau=\ln t$.

    The analytic derivative is stored alongside for cubic Hermite interpolation.
    """
    s = 2.0 / STANDARD_RADIAL_EXPONENT
    c, w = _disk_quadrature(64, 4.0)
    tau = np.linspace(np.log(t_lo), np.log(t_hi), n_tab)
    t = np.exp(tau)
    u = t[:, None] + c
    g_exp = np.sum(w * u ** (s - 1.0) / (-np.expm1(-u)), axis=-1)
    psi = np.log(g_exp) - (s - 1.0) * tau
    dpsi = -(t**s) / (-np.expm1(-t) * g_exp) + t - (s - 1.0)
    return float(tau[0]), float(tau[1] - tau[0]), psi, dpsi


_DISKBB_TAU0, _DISKBB_DTAU, _DISKBB_PSI, _DISKBB_DPSI = _diskbb_psi_table()


def _diskbb_log_shape(energy, tin):
    r"""Reduced log disk shape at $p=0.75$ by table interpolation.

    Same quantity as `_disk_quad_log_shape`. Beyond the tabulated $t \le 745$ the
    asymptotically linear transform is extrapolated.
    """
    # A traced index cannot gather from a NumPy table.
    psi_tab = jnp.asarray(_DISKBB_PSI)
    dpsi_tab = jnp.asarray(_DISKBB_DPSI)

    t = energy / tin
    t_safe = jnp.where(t > 0.0, t, 1.0)
    log_t = jnp.log(t_safe)
    x = (log_t - _DISKBB_TAU0) / _DISKBB_DTAU
    x_clip = jnp.clip(x, 0.0, _DISKBB_PSI.shape[0] - 1.0)
    i = jnp.clip(x_clip.astype(jnp.int32), 0, _DISKBB_PSI.shape[0] - 2)
    w = x_clip - i
    one_m_w = 1.0 - w
    psi = (
        (1.0 + 2.0 * w) * one_m_w**2 * psi_tab[i]
        + w * one_m_w**2 * _DISKBB_DTAU * dpsi_tab[i]
        + w**2 * (3.0 - 2.0 * w) * psi_tab[i + 1]
        + w**2 * (w - 1.0) * _DISKBB_DTAU * dpsi_tab[i + 1]
    )
    # Extend the asymptotically linear transform beyond the tabulated range.
    psi = psi + (x - x_clip) * _DISKBB_DTAU * jnp.where(x < 0.0, dpsi_tab[0], dpsi_tab[-1])
    return psi - t_safe + log_t, t > 0.0


def _is_standard_exponent(p) -> bool:
    """Whether ``p`` is the standard exponent as a Python float, hence fixed at trace time.

    The table carries no derivative in ``p``, so only a static exponent may use it;
    `Diskpbb` always passes an array.
    """
    return isinstance(p, float) and p == STANDARD_RADIAL_EXPONENT


def _disk_log_shape(energy, tin, p):
    """Reduced log disk shape and positive-energy mask, by table when ``p`` is static.

    ``p`` must arrive unconverted: an array selects the quadrature.
    """
    if _is_standard_exponent(p):
        return _diskbb_log_shape(energy, tin)
    return _disk_quad_log_shape(energy, tin, jnp.asarray(p))


def _disk_photon_flux(energy, tin, p):
    """Disk photon-flux density at unit XSPEC normalization."""
    log_shape, positive = _disk_log_shape(energy, tin, p)
    flux = 2.78e-3 * (STANDARD_RADIAL_EXPONENT / p) * tin**2 * jnp.exp(log_shape)
    return jnp.where(positive, flux, 0.0)


def _disk_log_band_shape(tin, p, e_min, e_max):
    """Reduced log disk shape integrated over ``[e_min, e_max]``, Gauss-Legendre in log energy."""
    tin = jnp.asarray(tin)
    lo, hi = jnp.log(e_min), jnp.log(e_max)
    energy = jnp.exp(0.5 * (hi - lo) * _DISK_GL_X + 0.5 * (hi + lo))
    log_weight = jnp.log(0.5 * (hi - lo) * _DISK_GL_W * energy)
    if not _is_standard_exponent(p):
        p = jnp.asarray(p)[..., None]
    log_shape, _ = _disk_log_shape(energy, tin[..., None], p)
    return logsumexp(log_weight + log_shape, axis=-1)


def _disk_band_photon_flux(tin, p, e_min, e_max):
    """Disk photon flux at unit XSPEC normalization over ``[e_min, e_max]``."""
    tin = jnp.asarray(tin)
    prefactor = 2.78e-3 * (STANDARD_RADIAL_EXPONENT / p) * tin**2
    return prefactor * jnp.exp(_disk_log_band_shape(tin, p, e_min, e_max))


def _normalized_disk_continuum(energy, tin, p, norm, flux_band):
    """Disk continuum whose photon flux over ``flux_band`` is ``norm``."""
    log_shape, positive = _disk_log_shape(energy, tin, p)
    log_band = _disk_log_band_shape(tin, p, *flux_band)
    flux = _exp_normalized(log_shape - log_band, norm)
    return jnp.where(positive, flux, 0.0)


def _xspec_normalization(tin, p, norm, flux_band):
    r"""XSPEC normalization $\cos i\,(r_{\rm in}/d)^2$ of a band-flux-normalized disk."""
    tin = jnp.asarray(tin)
    static = _is_standard_exponent(p)
    usable = (tin > 0.0) if static else (tin > 0.0) & (jnp.asarray(p) > 0.0)
    tin_safe = jnp.where(tin > 0.0, tin, 1.0)
    # A static exponent stays a float so the table remains selected.
    p_safe = p if static else jnp.where(jnp.asarray(p) > 0.0, p, STANDARD_RADIAL_EXPONENT)
    log_band = _disk_log_band_shape(tin_safe, p_safe, *flux_band)
    log_prefactor = (
        jnp.log(2.78e-3 * STANDARD_RADIAL_EXPONENT) - jnp.log(p_safe) + 2.0 * jnp.log(tin_safe)
    )
    converted = _exp_normalized(-(log_band + log_prefactor), norm)
    return jnp.where(usable, converted, 0.0)


def _validate_flux_band(flux_band: tuple[float, float]) -> tuple[float, float]:
    """Return ``flux_band`` as floats, rejecting empty, reversed or unbounded bands."""
    e_min, e_max = flux_band
    if not 0.0 < e_min < e_max < np.inf:
        raise ValueError(f"flux_band must satisfy 0 < e_min < e_max < inf, got {flux_band}")
    return float(e_min), float(e_max)


class Diskbb(AdditiveComponent):
    r"""Multicolor disk blackbody with radial exponent fixed at $p=0.75$.

    $$\mathcal{M}(E)=\frac{K}{\Phi(T_{\rm in})}\,
    \frac{2.78\times10^{-3}}{T_{\rm in}}
    \int_0^{T_{\rm in}}\left(\frac{T}{T_{\rm in}}\right)^{-11/3}
    \frac{E^2}{\exp(E/T)-1}\,\mathrm{d}T.$$

    ``norm`` is the unabsorbed photon flux over ``flux_band``. Fits publish the XSPEC
    normalization $\cos i\,(r_{\rm in}/d)^2$ as the derived quantity ``norm_xspec``.

    !!! abstract "Parameters"
        * $T_{\rm in}$ (``Tin``) $[\mathrm{keV}]$: inner-disk temperature
        * $K$ (``norm``) $[\mathrm{photons\,cm^{-2}\,s^{-1}}]$: photon flux over
          ``flux_band``

    !!! abstract "Constructor arguments"
        * ``flux_band``: normalization band in keV; defaults to 0.5--10 keV
    """

    def __init__(self, flux_band: tuple[float, float] = DEFAULT_FLUX_BAND):
        self.__flux_band = _validate_flux_band(flux_band)
        self.Tin = nnx.Param(1.0)
        self.norm = nnx.Param(1e-4)

    @property
    def _flux_band(self) -> tuple[float, float]:
        """Energy band, in keV, over which ``norm`` is the photon flux."""
        return self._flux_band

    def continuum(self, energy):
        return _normalized_disk_continuum(
            energy,
            jnp.asarray(self.Tin),
            STANDARD_RADIAL_EXPONENT,
            jnp.asarray(self.norm),
            self._flux_band,
        )

    def derived_quantities(self):
        return {
            "norm_xspec": _xspec_normalization(
                jnp.asarray(self.Tin),
                STANDARD_RADIAL_EXPONENT,
                jnp.asarray(self.norm),
                self._flux_band,
            )
        }


class Diskpbb(AdditiveComponent):
    r"""Multicolor disk blackbody with $T(r)\propto r^{-p}$.

    $$\mathcal{M}(E)=\frac{K}{\Phi(T_{\rm in},p)}\,
    \frac{2.78\times10^{-3}}{T_{\rm in}}\left(\frac{0.75}{p}\right)
    \int_0^{T_{\rm in}}\left(\frac{T}{T_{\rm in}}\right)^{-2/p-1}
    \frac{E^2}{\exp(E/T)-1}\,\mathrm{d}T.$$

    ``norm`` is the unabsorbed photon flux over ``flux_band``. Fits publish the XSPEC
    normalization $\cos i\,(r_{\rm in}/d)^2$ as the derived quantity ``norm_xspec``.

    !!! abstract "Parameters"
        * $T_{\rm in}$ (``Tin``) $[\mathrm{keV}]$: inner-disk temperature
        * $p$ (``p``): radial temperature exponent
        * $K$ (``norm``) $[\mathrm{photons\,cm^{-2}\,s^{-1}}]$: photon flux over
          ``flux_band``

    !!! abstract "Constructor arguments"
        * ``flux_band``: normalization band in keV; defaults to 0.5--10 keV
    """

    def __init__(self, flux_band: tuple[float, float] = DEFAULT_FLUX_BAND):
        self.__flux_band = _validate_flux_band(flux_band)
        self.Tin = nnx.Param(1.0)
        self.p = nnx.Param(STANDARD_RADIAL_EXPONENT)
        self.norm = nnx.Param(1e-4)

    @property
    def _flux_band(self) -> tuple[float, float]:
        """Energy band, in keV, over which ``norm`` is the photon flux."""
        return self._flux_band

    def continuum(self, energy):
        return _normalized_disk_continuum(
            energy,
            jnp.asarray(self.Tin),
            jnp.asarray(self.p),
            jnp.asarray(self.norm),
            self._flux_band,
        )

    def derived_quantities(self):
        return {
            "norm_xspec": _xspec_normalization(
                jnp.asarray(self.Tin),
                jnp.asarray(self.p),
                jnp.asarray(self.norm),
                self._flux_band,
            )
        }
