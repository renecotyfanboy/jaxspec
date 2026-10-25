# Spectral model base

::: jaxspec.model.abc.SpectralModel
    options:
      members:
        - from_component
        - from_string
        - from_dict
        - to_dict
        - graph
        - parameters
        - branches
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

::: jaxspec.model.abc.OperatorComponent
    options:
      members:
        - __call__
        - apply
      show_root_heading: true
      show_root_toc_entry: false

::: jaxspec.model.abc.ModelComponent
    options:
      members:
        - get_config
      show_root_heading: true
      show_root_toc_entry: false
