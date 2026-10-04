# OGIP input validation

Input handling must preserve the calibration before any model, convolution or
likelihood is evaluated. The public `Instrument.from_matrix`,
`Observation.from_matrix` and OGIP file loaders use the same physical units:
energies in keV, effective area in cm² and exposure in seconds. Plain arrays and
legacy columns without units use those defaults. Explicit Astropy/FITS units are
converted; incompatible units fail with the relevant field name.

The [OGIP response specification](https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html)
defines `F_CHAN` as the first detector **label** in a contiguous group, `N_CHAN`
as its length and `N_GRP` as the number of active groups. Consequently, a response
group beginning at channel 12 with two entries maps to labels 12 and 13 even if
the EBOUNDS rows are `[10, 12, 13]`. A group beginning at 10 with two entries is
invalid for those EBOUNDS: label 11 is absent. The loader rejects it instead of
clipping or shifting its weights. Actual EBOUNDS labels also allow older files
without the later `TLMIN` convention to remain unambiguous.

Fixed and variable-length compressed rows, scalar one-entry columns and empty
`N_GRP=0` rows are supported. Only active elements are interpreted; unused
fixed-width padding does not enter the response. Multiple MATRIX extensions
currently raise a clear unsupported-format error rather than discard all but
the first. This is a current implementation limit, not a claim that such files
violate OGIP.

The loader checks finite nonnegative active weights and areas, ordered photon-energy
bins, unique integer channel labels and a positive scalar exposure. Zero area,
zero lower energy edges and response columns whose sums differ from one remain
valid. It does not renormalize a supplied RMF. Without an ARF, the caller selects
the combined-RSP convention, which factors the supplied response into its area
and redistribution. Explicit area-valued MATRIX units or `HDUCLAS3=FULL` reject
an additional ARF because that would apply effective area twice. A unitless,
unclassified legacy file still requires the caller to know whether an ARF is
needed.

Nominal detector EBOUNDS are checked per channel and retain their original
label order. They need not increase with energy: RGS wavelength channels have
decreasing energy bounds. Sorting those rows by energy would misalign the PHA
and response. The ascending/nonoverlap rule applies to the incident photon
grid, not to the detector's labeling convention.

One explicit legacy unit convention has a narrow compatibility rule. The
[CXC source-catalog product documentation](https://cxc.cfa.harvard.edu/csc1/data_products/usage/)
labels ACIS redistribution matrices `au`. Only an `au` MATRIX with the complete
CHANDRA/ACIS, OGIP, RESPONSE/RSP_MATRIX/REDIST header context is interpreted as
dimensionless. The values and source file stay unchanged. `DataRMF` retains
`matrix_unit_original` and `compatibility_notes`; `Instrument.attrs` exposes
`response_matrix_unit_original` and `response_matrix_compatibility_notes`,
including the primary source URL. Other `au` contexts are rejected. This
exception never makes the response an effective-area matrix or replaces its ARF.

Adjacent calibration boundaries sometimes differ by storage roundoff. The
installed NuSTAR observation `10014001001` has 578 adjacent ARF edge pairs with
one float32-ULP overlap: at most 7.63e-6 keV, 1.19e-7 relative or 0.0191% of a
local bin width. Validation accepts overlap up to one relative machine epsilon
of the original input dtype, capped at 1% of either adjacent width. The values
themselves are retained exactly. Larger overlap, reversed bins and nonfinite
edges are rejected. Canonical float64 storage is constructed only after this
native-precision check.

Optional associated-file lookup still loads an existing named ARF, RMF or
background file. Optionality affects only a missing-file error. Lookup accepts
the exact name or its `.gz` counterpart; similarly prefixed backups are not
substituted. Simulated observations inherit the instrument's actual channel
labels. Observation configurations expose `e_min_channel` and `e_max_channel`
on every raw `instrument_channel`, aligned to the grouping matrix columns.
Rejected and excluded columns remain present with zero grouping weights, so
detector-background integration can preserve gaps inside a grouped bin.

Offline checks are in `test_ogip_units.py`, `test_ogip_rmf_groups.py`,
`test_response_validation.py`, `test_detector_channels.py`,
`test_ogip_paths.py`, `test_ogip_mock.py` and
`test_obsconf_detector_energies.py`. They include independent folded-count
calculations and deliberately malformed FITS fixtures. Before/after checks of
the installed NuSTAR response and bundled XMM PN, MOS1 and MOS2 observations
preserved every sparse response coordinate, response weight, effective area,
energy boundary, folded transfer value and observed count bit for bit.
