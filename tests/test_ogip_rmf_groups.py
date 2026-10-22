"""Independent compressed-response fixtures exercise detector labels and OGIP padding."""

import numpy as np
import pytest

from astropy.io import fits

from jaxspec.data import Instrument
from jaxspec.data.ogip import DataRMF


def write_response(
    path,
    *,
    channels=(10, 12, 13, 20),
    n_grp=(2, 0, 1),
    starts=((12, 20), (), (10,)),
    widths=((2, 1), (), (1,)),
    values=((0.25, 0.5, 0.125), (), (0.75,)),
    storage="variable",
    matrix_unit=None,
    full=False,
    extra_matrix=False,
    include_origin=True,
):
    """Write compressed rows with known masses; fixed storage has hostile unused padding."""
    rows = len(n_grp)
    group_format, value_format = "PJ()", "PD()"
    if storage == "fixed":
        max_group = max(1, max(len(row) for row in starts)) + 1
        max_values = max(1, max(len(row) for row in values)) + 1

        def padded(items, length, padding):
            """Fill unused slots with values that must not enter the active response."""
            result = np.full((rows, length), padding)
            for index, row in enumerate(items):
                result[index, : len(row)] = row
            return result

        starts = padded(starts, max_group, -999)
        widths = padded(widths, max_group, -999)
        values = padded(values, max_values, np.nan)
        group_format, value_format = f"{max_group}J", f"{max_values}D"
    elif storage == "scalar":
        starts = np.asarray(starts).reshape(rows)
        widths = np.asarray(widths).reshape(rows)
        values = np.asarray(values).reshape(rows)
        group_format, value_format = "J", "D"
    matrix = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ENERG_LO", format="D", unit="keV", array=np.arange(rows) + 1),
            fits.Column(name="ENERG_HI", format="D", unit="keV", array=np.arange(rows) + 2),
            fits.Column(name="N_GRP", format="J", array=n_grp),
            fits.Column(name="F_CHAN", format=group_format, array=starts),
            fits.Column(name="N_CHAN", format=group_format, array=widths),
            fits.Column(name="MATRIX", format=value_format, unit=matrix_unit, array=values),
        ],
        name="MATRIX",
    )
    if include_origin:
        matrix.header["TLMIN4"] = min(channels)
    if full:
        matrix.header["HDUCLAS3"] = "FULL"
    ebounds = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="J", array=channels),
            fits.Column(name="E_MIN", format="D", unit="keV", array=np.arange(len(channels)) + 1),
            fits.Column(name="E_MAX", format="D", unit="keV", array=np.arange(len(channels)) + 2),
        ],
        name="EBOUNDS",
    )
    extensions = [fits.PrimaryHDU(), matrix, ebounds]
    if extra_matrix:
        matrix.header["EXTVER"] = 1
        second = matrix.copy()
        second.header["EXTVER"] = 2
        extensions.append(second)
    fits.HDUList(extensions).writeto(path)
    return path


@pytest.mark.parametrize("storage", ["fixed", "variable"])
@pytest.mark.parametrize("include_origin", [False, True])
def test_groups_follow_labels_and_only_active_elements(tmp_path, storage, include_origin):
    response = DataRMF.from_file(
        write_response(tmp_path / "gaps.rmf", storage=storage, include_origin=include_origin)
    )
    expected = np.array([[0, 0, 0.75], [0.25, 0, 0], [0.5, 0, 0], [0.125, 0, 0]])
    np.testing.assert_array_equal(response.matrix, expected)
    np.testing.assert_array_equal(response.channel, [10, 12, 13, 20])
    assert response.sparse_matrix.nnz == 4


def test_scalar_single_value_columns(tmp_path):
    path = write_response(
        tmp_path / "scalar.rmf",
        channels=[7, 9],
        n_grp=[1, 1],
        starts=[[9], [7]],
        widths=[[1], [1]],
        values=[[0.25], [0.75]],
        storage="scalar",
    )
    np.testing.assert_array_equal(DataRMF.from_file(path).matrix, [[0, 0.75], [0.25, 0]])


def test_empty_response_rows_do_not_read_fixed_padding(tmp_path):
    path = write_response(
        tmp_path / "empty.rmf",
        channels=[1, 2],
        n_grp=[0],
        starts=[[]],
        widths=[[]],
        values=[[]],
        storage="fixed",
    )
    response = DataRMF.from_file(path)
    assert response.sparse_matrix.nnz == 0
    np.testing.assert_array_equal(response.matrix, np.zeros((2, 1)))


@pytest.mark.parametrize("start,width", [(11, 2), (10, 2), (20, 2), (9, 1)])
def test_group_interval_cannot_skip_missing_channel_labels(tmp_path, start, width):
    path = write_response(
        tmp_path / "undeclared.rmf",
        n_grp=[1],
        starts=[[start]],
        widths=[[width]],
        values=[np.ones(width)],
    )
    with pytest.raises(ValueError, match=r"RMF row 0.*absent from EBOUNDS"):
        DataRMF.from_file(path)


@pytest.mark.parametrize(
    "change,message",
    [
        ({"n_grp": [-1]}, "N_GRP"),
        ({"n_grp": [2]}, "group arrays"),
        ({"widths": [[0]]}, "N_CHAN"),
        ({"widths": [[5]]}, "N_CHAN"),
        ({"widths": [[2]], "starts": [[12]]}, "MATRIX values"),
        ({"values": [[-0.1]]}, "nonnegative"),
        ({"values": [[np.nan]]}, "finite"),
        ({"values": [[np.inf]]}, "finite"),
    ],
)
def test_invalid_active_groups_fail_before_sparsification(tmp_path, change, message):
    options = {"n_grp": [1], "starts": [[10]], "widths": [[1]], "values": [[0.5]]}
    options.update(change)
    with pytest.raises(ValueError, match=message):
        DataRMF.from_file(write_response(tmp_path / "invalid.rmf", **options))


def test_overlapping_groups_cannot_double_count_a_detector_channel(tmp_path):
    path = write_response(
        tmp_path / "overlap.rmf",
        n_grp=[2],
        starts=[[12, 13]],
        widths=[[2, 1]],
        values=[[0.25, 0.5, 0.75]],
    )
    with pytest.raises(ValueError, match="overlapping"):
        DataRMF.from_file(path)


def test_multiple_matrix_extensions_fail_explicitly(tmp_path):
    path = write_response(tmp_path / "multiple.rmf", extra_matrix=True)
    with pytest.raises(ValueError, match="multiple MATRIX"):
        DataRMF.from_file(path)


@pytest.mark.parametrize("storage", ["fixed", "variable"])
def test_complete_area_matrix_units_convert_before_rsp_factorization(tmp_path, storage):
    path = write_response(tmp_path / "area.rsp", matrix_unit="m2", storage=storage)
    response = Instrument.from_ogip_file(path)
    np.testing.assert_allclose(response.area, [8750, 0, 7500])
    np.testing.assert_allclose(
        response.redistribution.data.todense() * response.area.data,
        np.array([[0, 0, 0.75], [0.25, 0, 0], [0.5, 0, 0], [0.125, 0, 0]]) * 10000,
    )


@pytest.mark.parametrize("unit,full", [("cm2", False), (None, True)])
def test_an_arf_cannot_multiply_an_already_complete_response(tmp_path, unit, full):
    path = write_response(tmp_path / "area.rsp", matrix_unit=unit, full=full)
    # Failure must identify the double-area problem before trying to read an ARF.
    with pytest.raises(ValueError, match="already includes effective area"):
        Instrument.from_ogip_file(path, tmp_path / "unused.arf")


def test_incompatible_response_matrix_units_fail(tmp_path):
    path = write_response(tmp_path / "time.rmf", matrix_unit="s")
    with pytest.raises(ValueError, match="MATRIX units"):
        DataRMF.from_file(path)


def test_native_fits_edge_roundoff_survives_canonical_instrument_storage(tmp_path):
    path = write_response(tmp_path / "rounded.rsp", storage="fixed")
    low = np.array([1, np.nextafter(np.float32(2), np.float32(0)), 3], dtype=np.float32)
    with fits.open(path, mode="update") as hdus:
        original = hdus["MATRIX"]
        columns = [
            fits.Column(name="ENERG_LO", format="E", unit="keV", array=low),
            fits.Column(name="ENERG_HI", format="E", unit="keV", array=[2, 3, 4]),
            *list(original.columns)[2:],
        ]
        hdus["MATRIX"] = fits.BinTableHDU.from_columns(columns, name="MATRIX")
    response = Instrument.from_ogip_file(path)
    assert response.e_min_unfolded.dtype == np.float64
    np.testing.assert_array_equal(response.e_min_unfolded, low.astype(np.float64))


def test_grating_detector_bounds_can_decrease_in_channel_order(tmp_path):
    path = write_response(tmp_path / "grating.rsp")
    with fits.open(path, mode="update") as hdus:
        hdus["EBOUNDS"].data["E_MIN"] = [4, 3, 2, 1]
        hdus["EBOUNDS"].data["E_MAX"] = [5, 4, 3, 2]
    response = Instrument.from_ogip_file(path)
    np.testing.assert_array_equal(response.e_min_channel, [4, 3, 2, 1])
    np.testing.assert_array_equal(response.channel, [10, 12, 13, 20])
    np.testing.assert_allclose(
        response.redistribution.data.todense() * response.area.data,
        [[0, 0, 0.75], [0.25, 0, 0], [0.5, 0, 0], [0.125, 0, 0]],
    )
