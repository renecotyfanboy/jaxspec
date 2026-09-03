"""OGIP table models (ATable / MTable / ETable).

The fast tests validate the loader and the exact interpolation/redistribution
semantics against analytic expectations that were themselves verified live against
XSPEC 12.15.1 (multilinear corner weights, METHOD=1 log weights, additional-spectra
combination, redshift/escale energy scaling, LOELIMIT/HIELIMIT handling). The
``xspec``-marked tests re-run the comparison against live PyXSPEC on the same
synthesized files.
"""

import os
import re

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxspec.model.tabulated import ATable, ETable, MTable

DOC_PAGE = Path(__file__).resolve().parents[1] / "docs" / "examples" / "table_models.md"

REFLIONX = (
    "/Users/sdupourque/miniforge3/envs/ClusterXrayFluctuations/heasoft/spectral/modelData/"
    "reflionx.mod"
)


def write_table(
    path,
    *,
    param_specs,
    energ_lo,
    energ_hi,
    spectra,
    addsp=None,
    add_names=(),
    additive=True,
    redshift=False,
    escale=False,
    loelimit=None,
    hielimit=None,
    hduclas1="XSPEC TABLE MODEL",
):
    """Write a minimal OGIP 92-009 table-model FITS file.

    ``param_specs`` is a list of ``{"name", "method", "values", "initial"}`` dicts for
    the interpolation parameters; ``spectra`` (and each ``addsp`` entry) has one row
    per grid point in C order (last parameter varying fastest).
    """
    from astropy.io import fits

    addsp = addsp or []
    n_int, n_add = len(param_specs), len(add_names)
    max_vals = max(len(p["values"]) for p in param_specs)

    primary = fits.PrimaryHDU()
    primary.header["MODLNAME"] = "testtab"
    primary.header["MODLUNIT"] = "photons/cm^2/s" if additive else " "
    primary.header["REDSHIFT"] = bool(redshift)
    primary.header["ADDMODEL"] = bool(additive)
    primary.header["ESCALE"] = bool(escale)
    primary.header["HDUCLASS"] = "OGIP"
    primary.header["HDUCLAS1"] = hduclas1
    primary.header["HDUVERS1"] = "1.1.0"
    if loelimit is not None:
        primary.header["LOELIMIT"] = float(loelimit)
    if hielimit is not None:
        primary.header["HIELIMIT"] = float(hielimit)

    all_specs = list(param_specs) + [
        {"name": name, "method": 0, "values": [], "initial": 1.0} for name in add_names
    ]

    def column(name, fmt, values):
        return fits.Column(name=name, format=fmt, array=np.asarray(values))

    def pad(values):
        out = np.zeros(max_vals, dtype=np.float32)
        out[: len(values)] = values
        return out

    lows = [min(p["values"]) if p["values"] else 0.0 for p in all_specs]
    highs = [max(p["values"]) if p["values"] else 1e3 for p in all_specs]
    params_hdu = fits.BinTableHDU.from_columns(
        [
            column("NAME", "12A", [p["name"] for p in all_specs]),
            column("METHOD", "J", np.asarray([p["method"] for p in all_specs], dtype=np.int32)),
            column("INITIAL", "E", np.asarray([p["initial"] for p in all_specs], np.float32)),
            column("DELTA", "E", np.full(n_int + n_add, 0.01, dtype=np.float32)),
            column("MINIMUM", "E", np.asarray(lows, dtype=np.float32)),
            column("BOTTOM", "E", np.asarray(lows, dtype=np.float32)),
            column("TOP", "E", np.asarray(highs, dtype=np.float32)),
            column("MAXIMUM", "E", np.asarray(highs, dtype=np.float32)),
            column("NUMBVALS", "J", np.asarray([len(p["values"]) for p in all_specs], np.int32)),
            column("VALUE", f"{max_vals}E", np.stack([pad(p["values"]) for p in all_specs])),
        ],
        name="PARAMETERS",
    )
    params_hdu.header["NINTPARM"] = n_int
    params_hdu.header["NADDPARM"] = n_add

    energies_hdu = fits.BinTableHDU.from_columns(
        [
            column("ENERG_LO", "E", np.asarray(energ_lo, dtype=np.float32)),
            column("ENERG_HI", "E", np.asarray(energ_hi, dtype=np.float32)),
        ],
        name="ENERGIES",
    )

    grids = [np.asarray(p["values"], dtype=np.float64) for p in param_specs]
    mesh = np.meshgrid(*grids, indexing="ij")
    paramval = np.stack([m.ravel() for m in mesh], axis=-1).astype(np.float32)
    n_e = len(energ_lo)
    columns = [
        column("PARAMVAL", f"{n_int}E", paramval),
        column("INTPSPEC", f"{n_e}E", np.asarray(spectra, dtype=np.float32).reshape(-1, n_e)),
    ]
    for i, array in enumerate(addsp):
        columns.append(
            column(f"ADDSP{i + 1:03d}", f"{n_e}E", np.asarray(array, np.float32).reshape(-1, n_e))
        )
    spectra_hdu = fits.BinTableHDU.from_columns(columns, name="SPECTRA")

    fits.HDUList([primary, params_hdu, energies_hdu, spectra_hdu]).writeto(path, overwrite=True)
    return str(path)


def one_param(**kwargs):
    """Degenerate single-parameter grid with a flat unit response, for energy tests."""
    return write_table(
        param_specs=[{"name": "p", "method": 0, "values": [1.0, 2.0], "initial": 1.0}],
        spectra=kwargs.pop("spectra", [[1.0], [1.0]]),
        **kwargs,
    )


@pytest.fixture
def unit_line_table(tmp_path):
    """One table bin [1, 2] keV holding an integrated flux of 1.0."""
    return one_param(path=tmp_path / "unit.fits", energ_lo=[1.0], energ_hi=[2.0])


class TestATable:
    def test_redistribution_is_integral_preserving(self, unit_line_table):
        atable = ATable(unit_line_table)
        flux = atable.integrated_continuum(jnp.array([1.0, 1.5]), jnp.array([1.5, 2.0]))
        assert np.allclose(flux, [0.5, 0.5])
        flux = atable.integrated_continuum(jnp.array([0.5, 5.0]), jnp.array([1.5, 6.0]))
        assert np.allclose(flux, [0.5, 0.0])
        assert np.isclose(
            float(atable.integrated_continuum(jnp.array([0.0]), jnp.array([10.0]))[0]), 1.0
        )

    def test_norm_scales_flux(self, unit_line_table):
        atable = ATable(unit_line_table)
        atable.norm.set_value(jnp.asarray(3.0))
        flux = atable.integrated_continuum(jnp.array([1.0]), jnp.array([2.0]))
        assert np.isclose(float(flux[0]), 3.0)

    def test_multilinear_interpolation_mixed_methods(self, tmp_path):
        # Grids [1, 2] (linear) x [1, 10] (logarithmic); flat spectra with distinct
        # corner values. At a=1.5, b=sqrt(10) both weights are exactly 0.5, so the
        # result is the plain average of the four corners — the case verified live
        # against XSPEC 12.15.1.
        corners = {(0, 0): 1.0, (0, 1): 2.0, (1, 0): 5.0, (1, 1): 10.0}
        path = write_table(
            tmp_path / "bilinear.fits",
            param_specs=[
                {"name": "a", "method": 0, "values": [1.0, 2.0], "initial": 1.0},
                {"name": "b", "method": 1, "values": [1.0, 10.0], "initial": 1.0},
            ],
            energ_lo=[1.0],
            energ_hi=[2.0],
            spectra=[[corners[(i, j)]] for i in range(2) for j in range(2)],
        )
        atable = ATable(path)
        atable.a.set_value(jnp.asarray(1.5))
        atable.b.set_value(jnp.asarray(float(np.sqrt(10.0))))
        flux = atable.integrated_continuum(jnp.array([1.0]), jnp.array([2.0]))
        assert np.isclose(float(flux[0]), np.mean(list(corners.values())), rtol=1e-6)
        # Asymmetric weights pin the C-order (last param fastest) SPECTRA convention:
        # a transposed corner assignment passes the midpoint case above but not this.
        w_a, w_b = 0.25, 0.75
        atable.a.set_value(jnp.asarray(1.0 + w_a))
        atable.b.set_value(jnp.asarray(float(10.0**w_b)))
        flux = atable.integrated_continuum(jnp.array([1.0]), jnp.array([2.0]))
        expected = sum(
            corners[(i, j)] * (w_a if i else 1 - w_a) * (w_b if j else 1 - w_b)
            for i in range(2)
            for j in range(2)
        )
        assert np.isclose(float(flux[0]), expected, rtol=1e-6)

    def test_degenerate_single_value_grid(self, tmp_path):
        # NUMBVALS=1 freezes a dimension; the corner loop must keep both indices at 0.
        path = write_table(
            tmp_path / "single.fits",
            param_specs=[
                {"name": "frozen", "method": 0, "values": [7.0], "initial": 7.0},
                {"name": "p", "method": 0, "values": [1.0, 2.0], "initial": 1.0},
            ],
            energ_lo=[1.0],
            energ_hi=[2.0],
            spectra=[[2.0], [4.0]],
        )
        atable = ATable(path)
        atable.p.set_value(jnp.asarray(1.5))
        flux = atable.integrated_continuum(jnp.array([1.0]), jnp.array([2.0]))
        assert np.isclose(float(flux[0]), 3.0, rtol=1e-6)

    def test_redshift_and_escale_compose(self, tmp_path):
        # Verified live against XSPEC 12.15.1: the table is read at E*(1+z)/escale and
        # the additive flux keeps the 1/(1+z) factor — with z=0.5, escale=2, the unit
        # line at table [1, 2] keV lands on observed [4/3, 8/3] keV with total 1/1.5.
        path = one_param(
            path=tmp_path / "zesc.fits",
            energ_lo=[1.0],
            energ_hi=[2.0],
            redshift=True,
            escale=True,
        )
        atable = ATable(path)
        atable.z.set_value(jnp.asarray(0.5))
        atable.escale.set_value(jnp.asarray(2.0))
        flux = atable.integrated_continuum(jnp.array([4 / 3, 0.5]), jnp.array([8 / 3, 1.0]))
        assert np.allclose(flux, [1 / 1.5, 0.0], rtol=1e-6)

    def test_additional_parameters_combine_linearly(self, tmp_path):
        path = write_table(
            tmp_path / "addparm.fits",
            param_specs=[{"name": "p", "method": 0, "values": [1.0, 2.0], "initial": 1.0}],
            energ_lo=[1.0],
            energ_hi=[2.0],
            spectra=[[1.0], [1.0]],
            addsp=[[[0.5], [0.5]]],
            add_names=["q"],
        )
        atable = ATable(path)
        for q, expected in [(0.0, 1.0), (1.0, 1.5), (2.0, 2.0), (4.0, 3.0)]:
            atable.q.set_value(jnp.asarray(q))
            flux = atable.integrated_continuum(jnp.array([1.0]), jnp.array([2.0]))
            assert np.isclose(float(flux[0]), expected, rtol=1e-6)

    def test_redshift_shifts_and_time_dilates(self, tmp_path):
        path = one_param(path=tmp_path / "z.fits", energ_lo=[1.0], energ_hi=[2.0], redshift=True)
        atable = ATable(path)
        atable.z.set_value(jnp.asarray(1.0))
        flux = atable.integrated_continuum(jnp.array([0.5, 1.0]), jnp.array([1.0, 2.0]))
        assert np.allclose(flux, [0.5, 0.0])

    def test_escale_stretches_energies_without_flux_factor(self, tmp_path):
        path = one_param(path=tmp_path / "esc.fits", energ_lo=[1.0], energ_hi=[2.0], escale=True)
        atable = ATable(path)
        atable.escale.set_value(jnp.asarray(2.0))
        flux = atable.integrated_continuum(jnp.array([2.0, 1.0]), jnp.array([4.0, 2.0]))
        assert np.allclose(flux, [1.0, 0.0])

    def test_out_of_grid_parameters_clip_to_edge(self, tmp_path):
        path = write_table(
            tmp_path / "clip.fits",
            param_specs=[{"name": "p", "method": 0, "values": [1.0, 2.0], "initial": 1.0}],
            energ_lo=[1.0],
            energ_hi=[2.0],
            spectra=[[1.0], [3.0]],
        )
        atable = ATable(path)
        e_low, e_high = jnp.array([1.0]), jnp.array([2.0])
        atable.p.set_value(jnp.asarray(50.0))
        clipped = atable.integrated_continuum(e_low, e_high)
        atable.p.set_value(jnp.asarray(2.0))
        edge = atable.integrated_continuum(e_low, e_high)
        assert np.allclose(clipped, edge)
        gradient = jax.grad(
            lambda p: ATable(path)
            .photon_flux(e_low, e_high, params={"atable_1.p": p, "atable_1.norm": 1.0})
            .sum()
        )(1.5)
        assert np.isfinite(gradient) and gradient > 0


class TestMTable:
    def test_width_weighted_average_and_limits(self, tmp_path):
        path = write_table(
            tmp_path / "mt.fits",
            param_specs=[{"name": "p", "method": 0, "values": [1.0, 2.0], "initial": 1.0}],
            energ_lo=[1.0, 2.0],
            energ_hi=[2.0, 3.0],
            spectra=[[3.0, 5.0], [3.0, 5.0]],
            additive=False,
            loelimit=0.25,
            hielimit=0.75,
        )
        mtable = MTable(path)
        factor = mtable._factor(jnp.array([1.5, 0.1, 5.0]), jnp.array([2.5, 0.5, 6.0]))
        assert np.allclose(factor, [4.0, 0.25, 0.75])

    def test_out_of_range_defaults_to_one(self, tmp_path):
        path = one_param(path=tmp_path / "mt1.fits", energ_lo=[1.0], energ_hi=[2.0], additive=False)
        mtable = MTable(path)
        factor = mtable._factor(jnp.array([5.0, 0.1]), jnp.array([6.0, 0.5]))
        assert np.allclose(factor, [1.0, 1.0])

    def test_collapsed_bins_are_killed(self, tmp_path):
        path = one_param(path=tmp_path / "mt2.fits", energ_lo=[1.0], energ_hi=[2.0], additive=False)
        mtable = MTable(path)
        factor = mtable._factor(jnp.array([1.5, 1e-6]), jnp.array([1.6, 1e-6]))
        assert factor[1] == 0.0

    def test_pointwise_factor_matches_table_bins(self, tmp_path):
        path = write_table(
            tmp_path / "mt3.fits",
            param_specs=[{"name": "p", "method": 0, "values": [1.0, 2.0], "initial": 1.0}],
            energ_lo=[1.0, 2.0],
            energ_hi=[2.0, 3.0],
            spectra=[[3.0, 5.0], [3.0, 5.0]],
            additive=False,
        )
        mtable = MTable(path)
        factor = mtable.factor(jnp.array([1.5, 2.5, 0.5, 5.0]))
        assert np.allclose(factor, [3.0, 5.0, 1.0, 1.0])

    def test_redshift_shifts_without_flux_factor(self, tmp_path):
        # Verified live: multiplicative tables shift with z but never pick up 1/(1+z).
        path = one_param(
            path=tmp_path / "mtz.fits",
            energ_lo=[1.0],
            energ_hi=[2.0],
            spectra=[[3.0], [3.0]],
            additive=False,
            redshift=True,
        )
        mtable = MTable(path)
        mtable.z.set_value(jnp.asarray(1.0))
        factor = mtable._factor(jnp.array([0.5, 1.5]), jnp.array([1.0, 2.0]))
        assert np.allclose(factor, [3.0, 1.0])

    def test_escale_shifts_factor_pattern(self, tmp_path):
        path = one_param(
            path=tmp_path / "mtesc.fits",
            energ_lo=[1.0],
            energ_hi=[2.0],
            spectra=[[3.0], [3.0]],
            additive=False,
            escale=True,
        )
        mtable = MTable(path)
        mtable.escale.set_value(jnp.asarray(2.0))
        factor = mtable._factor(jnp.array([2.0, 0.5]), jnp.array([4.0, 1.0]))
        assert np.allclose(factor, [3.0, 1.0])


class TestETable:
    def test_exponential_of_combined_value(self, tmp_path):
        path = one_param(
            path=tmp_path / "et.fits",
            energ_lo=[1.0],
            energ_hi=[2.0],
            additive=False,
            loelimit=2.0,
        )
        etable = ETable(path)
        factor = etable._factor(jnp.array([1.0, 0.1]), jnp.array([2.0, 0.5]))
        # LOELIMIT is a final factor, never exponentiated.
        assert np.allclose(factor, [np.exp(-1.0), 2.0])

    def test_exponential_applied_after_width_average(self, tmp_path):
        # Verified live against XSPEC 12.15.1: a bin straddling optical depths 1 and 3
        # gives exp(-avg) = exp(-2) = 0.1353, not avg(exp) = 0.2088.
        path = write_table(
            tmp_path / "et2.fits",
            param_specs=[{"name": "p", "method": 0, "values": [1.0, 2.0], "initial": 1.0}],
            energ_lo=[1.0, 2.0],
            energ_hi=[2.0, 3.0],
            spectra=[[1.0, 3.0], [1.0, 3.0]],
            additive=False,
        )
        etable = ETable(path)
        factor = etable._factor(jnp.array([1.5]), jnp.array([2.5]))
        assert np.isclose(float(factor[0]), np.exp(-2.0), rtol=1e-6)

    def test_additional_parameters_combine_before_exponential(self, tmp_path):
        path = write_table(
            tmp_path / "et3.fits",
            param_specs=[{"name": "p", "method": 0, "values": [1.0, 2.0], "initial": 1.0}],
            energ_lo=[1.0],
            energ_hi=[2.0],
            spectra=[[1.0], [1.0]],
            addsp=[[[0.5], [0.5]]],
            add_names=["q"],
            additive=False,
        )
        etable = ETable(path)
        etable.q.set_value(jnp.asarray(2.0))
        factor = etable._factor(jnp.array([1.0]), jnp.array([2.0]))
        assert np.isclose(float(factor[0]), np.exp(-2.0), rtol=1e-6)


class TestLoader:
    def test_addmodel_mismatch_raises(self, unit_line_table):
        with pytest.raises(ValueError, match="ATable"):
            MTable(unit_line_table)

    def test_not_a_table_model_raises(self, tmp_path):
        path = one_param(
            path=tmp_path / "bad.fits", energ_lo=[1.0], energ_hi=[2.0], hduclas1="RESPONSE"
        )
        with pytest.raises(ValueError, match="OGIP"):
            ATable(path)

    def test_non_contiguous_energies_raise(self, tmp_path):
        path = one_param(
            path=tmp_path / "gap.fits",
            energ_lo=[1.0, 3.0],
            energ_hi=[2.0, 4.0],
            spectra=[[1.0, 1.0], [1.0, 1.0]],
        )
        with pytest.raises(ValueError, match="contiguous"):
            ATable(path)

    def test_name_sanitization_and_collisions(self, tmp_path):
        path = write_table(
            tmp_path / "names.fits",
            param_specs=[
                {"name": "Fe/solar", "method": 0, "values": [1.0, 2.0], "initial": 1.0},
                {"name": "log T", "method": 0, "values": [1.0, 2.0], "initial": 1.0},
                {"name": "lambda", "method": 0, "values": [1.0, 2.0], "initial": 1.0},
                {"name": "norm", "method": 0, "values": [1.0, 2.0], "initial": 1.0},
            ],
            energ_lo=[1.0],
            energ_hi=[2.0],
            spectra=[[1.0]] * 16,
        )
        atable = ATable(path)
        assert list(atable.table_parameters) == ["Fe_solar", "log_T", "lambda_", "norm_2"]
        assert atable.table_parameters["Fe_solar"] == "Fe/solar"
        # The added norm parameter must survive the collision untouched.
        assert float(atable.norm) == 1.0

    def test_parameter_names_cannot_shadow_component_api(self, tmp_path):
        # A FITS parameter named like a component attribute ('type', 'continuum',
        # 'factor', ...) must be renamed, or setattr would shadow the class API and
        # crash evaluation (or silently mistype the graph node).
        path = write_table(
            tmp_path / "shadow.fits",
            param_specs=[
                {"name": "type", "method": 0, "values": [1.0, 2.0], "initial": 1.0},
                {"name": "continuum", "method": 0, "values": [1.0, 2.0], "initial": 1.0},
            ],
            energ_lo=[1.0],
            energ_hi=[2.0],
            spectra=[[1.0]] * 4,
        )
        atable = ATable(path)
        assert atable.type == "additive"
        assert callable(atable.continuum)
        names = list(atable.table_parameters)
        flux = atable.photon_flux(
            jnp.array([1.0]),
            jnp.array([2.0]),
            params={f"atable_1.{name}": 1.5 for name in names} | {"atable_1.norm": 1.0},
        )
        assert np.isclose(float(flux[0]), 1.0, rtol=1e-6)

        mpath = write_table(
            tmp_path / "shadow_m.fits",
            param_specs=[{"name": "factor", "method": 0, "values": [1.0, 2.0], "initial": 1.0}],
            energ_lo=[1.0],
            energ_hi=[2.0],
            spectra=[[3.0], [3.0]],
            additive=False,
        )
        mtable = MTable(mpath)
        assert callable(mtable.factor)
        assert np.isclose(float(mtable.factor(jnp.array([1.5]))[0]), 3.0)

    @pytest.mark.parametrize(
        ("corrupt", "match"),
        [
            (lambda h: h[0].header.set("NXFLTEXP", 2), "NXFLTEXP"),
            (lambda h: h[0].header.set("NNPTFILE", "emulator.pt"), "neural-network"),
            (lambda h: h["PARAMETERS"].header.set("NINTPARM", 2), "PARAMETERS has"),
            (lambda h: h["PARAMETERS"].data["NUMBVALS"].__setitem__(0, 5), "NUMBVALS"),
        ],
        ids=["nxfltexp", "nnptfile", "param-count", "numbvals-mismatch"],
    )
    def test_loader_validation_on_corrupted_files(self, tmp_path, corrupt, match):
        from astropy.io import fits

        path = one_param(path=tmp_path / "valid.fits", energ_lo=[1.0], energ_hi=[2.0])
        with fits.open(path) as hdul:
            corrupt(hdul)
            hdul.writeto(tmp_path / "corrupted.fits", overwrite=True)
        with pytest.raises((ValueError, NotImplementedError), match=match):
            ATable(tmp_path / "corrupted.fits")

    def test_loader_validation_on_bad_grids(self, tmp_path):
        with pytest.raises(ValueError, match="strictly increasing"):
            ATable(
                write_table(
                    tmp_path / "decreasing.fits",
                    param_specs=[{"name": "p", "method": 0, "values": [2.0, 1.0], "initial": 1.0}],
                    energ_lo=[1.0],
                    energ_hi=[2.0],
                    spectra=[[1.0], [1.0]],
                )
            )
        with pytest.raises(ValueError, match="logarithmic"):
            ATable(
                write_table(
                    tmp_path / "logzero.fits",
                    param_specs=[{"name": "p", "method": 1, "values": [0.0, 1.0], "initial": 0.5}],
                    energ_lo=[1.0],
                    energ_hi=[2.0],
                    spectra=[[1.0], [1.0]],
                )
            )

    def test_spectra_row_count_mismatch_raises(self, tmp_path):
        from astropy.io import fits

        path = one_param(path=tmp_path / "valid.fits", energ_lo=[1.0], energ_hi=[2.0])
        with fits.open(path) as hdul:
            truncated = fits.BinTableHDU(data=hdul["SPECTRA"].data[:-1], name="SPECTRA")
            fits.HDUList([hdul[0], hdul["PARAMETERS"], hdul["ENERGIES"], truncated]).writeto(
                tmp_path / "truncated.fits", overwrite=True
            )
        with pytest.raises(ValueError, match="SPECTRA has"):
            ATable(tmp_path / "truncated.fits")

    def test_energy_band_without_overlap_raises(self, unit_line_table):
        with pytest.raises(ValueError, match="does not overlap"):
            ATable(unit_line_table, energy_band=(100.0, 200.0))

    def test_regenerated_file_is_reloaded(self, tmp_path):
        # The cache keys on (path, mtime, size): overwriting the file at the same path
        # in a live session must not serve the stale table.
        path = tmp_path / "regen.fits"
        one_param(path=path, energ_lo=[1.0], energ_hi=[2.0], spectra=[[1.0], [1.0]])
        first = float(ATable(path).integrated_continuum(jnp.array([1.0]), jnp.array([2.0]))[0])
        os.utime(path)  # guard against sub-resolution mtime on fast filesystems
        one_param(path=path, energ_lo=[1.0], energ_hi=[2.0], spectra=[[5.0], [5.0]])
        second = float(ATable(path).integrated_continuum(jnp.array([1.0]), jnp.array([2.0]))[0])
        assert first == 1.0 and second == 5.0

    def test_energy_band_restriction_matches_full_table(self, tmp_path):
        edges = np.geomspace(0.1, 50.0, 101)
        rng = np.random.default_rng(7)
        spectrum = rng.random(100)
        path = write_table(
            tmp_path / "band.fits",
            param_specs=[{"name": "p", "method": 0, "values": [1.0, 2.0], "initial": 1.0}],
            energ_lo=edges[:-1],
            energ_hi=edges[1:],
            spectra=[spectrum, spectrum],
        )
        e_low = jnp.geomspace(0.5, 10.0, 40)[:-1]
        e_high = jnp.geomspace(0.5, 10.0, 40)[1:]
        full = ATable(path).integrated_continuum(e_low, e_high)
        cropped = ATable(path, energy_band=(0.4, 12.0)).integrated_continuum(e_low, e_high)
        assert np.allclose(full, cropped, rtol=1e-6)
        assert ATable(path, energy_band=(0.4, 12.0))._table.e_edges.size < edges.size

    def test_instances_share_table_arrays(self, unit_line_table):
        first, second = ATable(unit_line_table), ATable(unit_line_table)
        assert first._table is second._table


class TestIntegration:
    def test_nnx_clone_shares_table(self, unit_line_table):
        from flax import nnx

        atable = ATable(unit_line_table)
        clone = nnx.clone(atable)
        assert clone._table is atable._table
        params = nnx.split(atable, nnx.Param, ...)[1]
        assert {path[0] for path, _ in nnx.to_flat_state(params)} == {"p", "norm"}

    def test_composition_and_flux_func(self, tmp_path, unit_line_table):
        from jaxspec.model.multiplicative import Tbabs

        mtable_path = one_param(
            path=tmp_path / "mt.fits",
            energ_lo=[0.1],
            energ_hi=[100.0],
            additive=False,
            spectra=[[0.5], [0.5]],
        )
        model = Tbabs() * MTable(mtable_path) * ATable(unit_line_table)
        e_low = jnp.geomspace(0.5, 10.0, 50)[:-1]
        e_high = jnp.geomspace(0.5, 10.0, 50)[1:]
        flux = model.photon_flux(
            e_low,
            e_high,
            params={
                "tbabs_1.nh": 0.0,
                "mtable_1.p": 1.0,
                "atable_1.p": 1.0,
                "atable_1.norm": 1.0,
            },
        )
        assert np.all(np.isfinite(flux))
        # Tbabs is transparent at nh=0, so the mtable factor 0.5 halves the table line.
        # rtol covers the base class's 2-point log-trapezoid quadrature of Tbabs.
        assert np.isclose(float(flux.sum()), 0.5, rtol=1e-3)

    def test_vmap_over_parameters(self, tmp_path):
        path = write_table(
            tmp_path / "vmap.fits",
            param_specs=[{"name": "p", "method": 0, "values": [1.0, 2.0], "initial": 1.0}],
            energ_lo=[1.0],
            energ_hi=[2.0],
            spectra=[[1.0], [3.0]],
        )
        atable = ATable(path)
        e_low, e_high = jnp.array([1.0]), jnp.array([2.0])

        def total(p):
            return atable.photon_flux(
                e_low, e_high, params={"atable_1.p": p, "atable_1.norm": 1.0}
            ).sum()

        totals = jax.vmap(total)(jnp.array([1.0, 1.5, 2.0]))
        assert np.allclose(totals, [1.0, 2.0, 3.0], rtol=1e-6)

    def test_fakeit_smoke(self, obsconfs, tmp_path):
        from jaxspec.data.util import fakeit_for_multiple_parameters

        edges = np.geomspace(0.1, 50.0, 201)
        path = write_table(
            tmp_path / "fakeit.fits",
            param_specs=[{"name": "slope", "method": 0, "values": [1.0, 3.0], "initial": 2.0}],
            energ_lo=edges[:-1],
            energ_hi=edges[1:],
            spectra=[np.diff(edges) * edges[:-1] ** -s for s in (1.0, 3.0)],
        )
        obsconf = obsconfs[0]
        parameters = {
            "atable_1.slope": jnp.linspace(1.2, 2.8, 4),
            "atable_1.norm": jnp.full(4, 1.0),
        }
        spectra = fakeit_for_multiple_parameters(obsconf, ATable(path), parameters)
        assert spectra.shape[0] == 4
        assert np.all(np.isfinite(spectra))


@pytest.mark.skipif(not DOC_PAGE.exists(), reason="documentation page not present")
def test_documented_table_building_snippet(tmp_path, monkeypatch):
    """The table-writing snippet published in the docs must keep working.

    Documentation code rots silently, and this one is a how-to users copy verbatim
    to turn their own model grid into an OGIP file: execute it as published and
    check the result reproduces the analytic power-law integral it advertises.
    """
    blocks = re.findall(r"```python\n(.*?)```", DOC_PAGE.read_text(), re.S)
    snippets = [block for block in blocks if "writeto" in block]
    assert len(snippets) == 1, "expected exactly one table-writing snippet in the docs"

    monkeypatch.chdir(tmp_path)
    exec(compile(snippets[0], str(DOC_PAGE), "exec"), {})

    atable = ATable(tmp_path / "mymodel.mod")
    assert list(atable.table_parameters) == ["slope"]

    energy = np.geomspace(0.5, 10.0, 200)
    flux = np.asarray(
        atable.photon_flux(
            jnp.asarray(energy[:-1]),
            jnp.asarray(energy[1:]),
            params={"atable_1.slope": 2.0, "atable_1.norm": 1.0},
        )
    )
    exact = -np.diff(energy**-1.0)  # integral of E**-2 over each bin
    # Residual is the table's own resolution (rebinning assumes flux is uniform
    # inside a tabulated bin), not the interpolation — see the page's closing note.
    assert np.median(np.abs(flux / exact - 1)) < 5e-3


@pytest.mark.skipif(not os.path.exists(REFLIONX), reason="local HEASOFT reflionx.mod not found")
def test_reflionx_real_table():
    atable = ATable(REFLIONX)
    assert list(atable.table_parameters) == ["Fe_solar", "Gamma", "Xi"]
    assert hasattr(atable, "z") and hasattr(atable, "norm")
    energy = jnp.geomspace(0.3, 10.0, 300)
    parameters = {
        "atable_1.Fe_solar": 1.0,
        "atable_1.Gamma": 2.0,
        "atable_1.Xi": 300.0,
        "atable_1.z": 0.05,
        "atable_1.norm": 1.0,
    }
    flux = atable.photon_flux(energy[:-1], energy[1:], params=parameters)
    assert np.all(np.isfinite(flux)) and float(flux.sum()) > 0
    gradient = jax.grad(
        lambda xi: atable.photon_flux(
            energy[:-1], energy[1:], params={**parameters, "atable_1.Xi": xi}
        ).sum()
    )(300.0)
    assert np.isfinite(gradient)


# --- Live XSPEC comparisons (run via `bash scripts/run_xspec_tests.sh`) -------------


def _xspec_table_flux(expression, values, e_min, e_max, n_bins):
    import xspec

    xspec.AllModels.clear()
    xspec.AllModels.setEnergies(f"{e_min} {e_max} {n_bins} log")
    model = xspec.Model(expression)
    model.setPars(*values)
    return np.asarray(model.values(0), dtype=np.float64)


@pytest.mark.xspec
def test_atable_vs_live_xspec(tmp_path):
    pytest.importorskip("xspec")
    edges = np.geomspace(0.1, 30.0, 201)
    grid_a, grid_b = [1.0, 1.5, 2.0], [1.0, 10.0, 100.0]
    spectra = [np.diff(edges) * edges[:-1] ** (-a) * np.log10(b) for a in grid_a for b in grid_b]
    path = write_table(
        tmp_path / "cmp_atable.fits",
        param_specs=[
            {"name": "alpha", "method": 0, "values": grid_a, "initial": 1.0},
            {"name": "xi", "method": 1, "values": grid_b, "initial": 10.0},
        ],
        energ_lo=edges[:-1],
        energ_hi=edges[1:],
        spectra=spectra,
        redshift=True,
    )
    e_low = jnp.asarray(np.geomspace(0.5, 9.0, 121)[:-1])
    e_high = jnp.asarray(np.geomspace(0.5, 9.0, 121)[1:])
    for alpha, xi, z in [(1.3, 4.0, 0.0), (1.9, 55.0, 0.12)]:
        reference = _xspec_table_flux(f"atable{{{path}}}", [alpha, xi, z, 1.0], 0.5, 9.0, 120)
        flux = np.asarray(
            ATable(path).photon_flux(
                e_low,
                e_high,
                params={
                    "atable_1.alpha": alpha,
                    "atable_1.xi": xi,
                    "atable_1.z": z,
                    "atable_1.norm": 1.0,
                },
            )
        )
        mask = reference > reference.max() * 1e-6
        assert np.median(np.abs(flux[mask] / reference[mask] - 1)) < 1e-4


@pytest.mark.xspec
def test_mtable_vs_live_xspec(tmp_path):
    pytest.importorskip("xspec")
    edges = np.geomspace(0.1, 30.0, 201)
    factors = [np.exp(-t / edges[:-1]) for t in (0.1, 1.0, 5.0)]
    path = write_table(
        tmp_path / "cmp_mtable.fits",
        param_specs=[{"name": "tau", "method": 0, "values": [0.1, 1.0, 5.0], "initial": 1.0}],
        energ_lo=edges[:-1],
        energ_hi=edges[1:],
        spectra=factors,
        additive=False,
    )
    # Probe the factor with a flat frozen powerlaw, like tests/xspec_utils.py does.
    reference = _xspec_table_flux(f"mtable{{{path}}}*powerlaw", [0.7, 0.0, 1.0], 0.5, 9.0, 120)
    e_low = jnp.asarray(np.geomspace(0.5, 9.0, 121)[:-1])
    e_high = jnp.asarray(np.geomspace(0.5, 9.0, 121)[1:])
    from jaxspec.model.additive import Powerlaw

    flux = np.asarray(
        (MTable(path) * Powerlaw()).photon_flux(
            e_low,
            e_high,
            params={"mtable_1.tau": 0.7, "powerlaw_1.alpha": 0.0, "powerlaw_1.norm": 1.0},
        )
    )
    mask = reference > reference.max() * 1e-6
    assert np.median(np.abs(flux[mask] / reference[mask] - 1)) < 1e-3


@pytest.mark.xspec
def test_etable_vs_live_xspec(tmp_path):
    pytest.importorskip("xspec")
    edges = np.geomspace(0.1, 30.0, 201)
    depths = [t / edges[:-1] for t in (0.1, 1.0, 5.0)]
    path = write_table(
        tmp_path / "cmp_etable.fits",
        param_specs=[{"name": "tau", "method": 0, "values": [0.1, 1.0, 5.0], "initial": 1.0}],
        energ_lo=edges[:-1],
        energ_hi=edges[1:],
        spectra=depths,
        additive=False,
    )
    reference = _xspec_table_flux(f"etable{{{path}}}*powerlaw", [0.7, 0.0, 1.0], 0.5, 9.0, 120)
    e_low = jnp.asarray(np.geomspace(0.5, 9.0, 121)[:-1])
    e_high = jnp.asarray(np.geomspace(0.5, 9.0, 121)[1:])
    from jaxspec.model.additive import Powerlaw

    flux = np.asarray(
        (ETable(path) * Powerlaw()).photon_flux(
            e_low,
            e_high,
            params={"etable_1.tau": 0.7, "powerlaw_1.alpha": 0.0, "powerlaw_1.norm": 1.0},
        )
    )
    mask = reference > reference.max() * 1e-6
    assert np.median(np.abs(flux[mask] / reference[mask] - 1)) < 1e-3


@pytest.mark.xspec
def test_redshift_escale_composition_vs_live_xspec(tmp_path):
    pytest.importorskip("xspec")
    edges = np.geomspace(0.1, 30.0, 201)
    path = write_table(
        tmp_path / "cmp_zesc.fits",
        param_specs=[{"name": "slope", "method": 0, "values": [1.0, 3.0], "initial": 2.0}],
        energ_lo=edges[:-1],
        energ_hi=edges[1:],
        spectra=[np.diff(edges) * edges[:-1] ** -s for s in (1.0, 3.0)],
        redshift=True,
        escale=True,
    )
    # Parameter order in the file: slope, Escale, z, norm.
    reference = _xspec_table_flux(f"atable{{{path}}}", [1.8, 2.0, 0.5, 1.0], 0.5, 9.0, 120)
    e_low = jnp.asarray(np.geomspace(0.5, 9.0, 121)[:-1])
    e_high = jnp.asarray(np.geomspace(0.5, 9.0, 121)[1:])
    flux = np.asarray(
        ATable(path).photon_flux(
            e_low,
            e_high,
            params={
                "atable_1.slope": 1.8,
                "atable_1.escale": 2.0,
                "atable_1.z": 0.5,
                "atable_1.norm": 1.0,
            },
        )
    )
    mask = reference > reference.max() * 1e-6
    assert np.median(np.abs(flux[mask] / reference[mask] - 1)) < 1e-4


@pytest.mark.xspec
@pytest.mark.skipif(not os.path.exists(REFLIONX), reason="local HEASOFT reflionx.mod not found")
def test_reflionx_vs_live_xspec():
    pytest.importorskip("xspec")
    e_low = jnp.asarray(np.geomspace(1.0, 9.0, 121)[:-1])
    e_high = jnp.asarray(np.geomspace(1.0, 9.0, 121)[1:])
    for fe, gamma, xi in [(1.0, 2.0, 300.0), (2.5, 1.7, 40.0)]:
        reference = _xspec_table_flux(
            f"atable{{{REFLIONX}}}", [fe, gamma, xi, 0.0, 1.0], 1.0, 9.0, 120
        )
        flux = np.asarray(
            ATable(REFLIONX).photon_flux(
                e_low,
                e_high,
                params={
                    "atable_1.Fe_solar": fe,
                    "atable_1.Gamma": gamma,
                    "atable_1.Xi": xi,
                    "atable_1.z": 0.0,
                    "atable_1.norm": 1.0,
                },
            )
        )
        mask = reference > reference.max() * 1e-6
        assert np.median(np.abs(flux[mask] / reference[mask] - 1)) < 1e-3
