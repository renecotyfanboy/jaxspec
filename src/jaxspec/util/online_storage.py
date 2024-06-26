import pooch

table_manager = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("jaxspec"),
    base_url="https://github.com/renecotyfanboy/jaxspec-database/raw/main/",
    # The registry specifies the files that can be fetched
    registry={
        "abundances.dat": "sha256:6a7826331f0de308af4631eed5c3b65accda99cd1aa8766f54119dd285b57992",
        "apec.nc": "sha256:52e10e1e4147453890dac68845a1a629954283579eac602419634d43d3c101f9",
        "xsect_tbabs_wilm.fits": "sha256:3cf45e45c9d671c4c4fc128314b7c3a68b30f096eede6b3eb08bf55224a44935",
        "xsect_phabs_aspl.fits": "sha256:3eaffba2a62e3a611e0a4e1ff4a57342d7d576f023d7bbb632710dc75b9a5019",
        "xsect_wabs_angr.fits": "sha256:9b3073a477a30b52e207f2c4bf79afc6ae19abba8f207190ac4c697024f74073",
    },
)
