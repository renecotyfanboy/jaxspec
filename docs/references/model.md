# Spectral model base

::: jaxspec.model.abc.SpectralModel
    options:
      members:
        - from_component
        - photon_flux
        - energy_flux
        - integrated_flux
      show_root_heading: true
      show_root_toc_entry: false

::: jaxspec.model.abc.AdditiveComponent
    options:
      members:
        - continuum
        - integrated_continuum
      show_root_heading: true
      show_root_toc_entry: false

::: jaxspec.model.abc.MultiplicativeComponent
    options:
      members:
        - factor
      show_root_heading: true
      show_root_toc_entry: false

::: jaxspec.model.abc.HasDerivedQuantities
    options:
      members:
        - derived_quantities
      show_root_heading: true
      show_root_toc_entry: false