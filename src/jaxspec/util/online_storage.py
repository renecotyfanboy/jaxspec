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
        "nsatmosdata.fits": "sha256:faca712f87710ecb866b4ab61be593a6813517c44f6e8e92d689b38d42e1b6dc",
        "example_data/NGC7793_ULX4/MOS2background_spectrum.fits": "sha256:5387923be0bf39229f4390dd5e85095a3d534b43a69d6d3179b832ebb366d173",
        "example_data/NGC7793_ULX4/MOS1background_spectrum.fits": "sha256:265fd7465fb1a355f915d9902443ba2fd2be9aface04723056a8376971e3cf14",
        "example_data/NGC7793_ULX4/MOS2.rmf": "sha256:b6af00603dece33dcda35d093451c947059af2e1e45c31c5a0ffa223b7fb693d",
        "example_data/NGC7793_ULX4/PN.arf": "sha256:0ee897a63b6de80589c2da758d7477c54ba601b788bf32d4d16bbffa839acb73",
        "example_data/NGC7793_ULX4/MOS1.rmf": "sha256:2d1138d22c31c5398a4eed1170b0b88b07350ecfec7a7aef4f550871cb4309ae",
        "example_data/NGC7793_ULX4/PN_spectrum_grp20.fits": "sha256:a985e06076bf060d3a5331f20413afa9208a8d771fa6c671e8918a1860577c90",
        "example_data/NGC7793_ULX4/MOS2_spectrum_grp.fits": "sha256:dccc7eda9d3d2e4aac2af4ca13d41ab4acc621265004d50a1586a187f7a04ffc",
        "example_data/NGC7793_ULX4/MOS1_spectrum_grp.fits": "sha256:7e1ff664545bab4fdce1ef94768715b4d87a39b252b61e070e71427e5c8692ac",
        "example_data/NGC7793_ULX4/MOS1.arf": "sha256:9017ada6a391d46f9b569b8d0338fbabb62a5397e7c29eb0a16e4e02d4868159",
        "example_data/NGC7793_ULX4/PN.rmf": "sha256:91ba9ef82da8b9f73e6a799dfe097b87c68a7020ac6c5aa0dcd4067bf9cb4287",
        "example_data/NGC7793_ULX4/MOS2.arf": "sha256:a126ff5a95a5f4bb93ed846944cf411d6e1c448626cb73d347e33324663d8b3f",
        "example_data/NGC7793_ULX4/PNbackground_spectrum.fits": "sha256:55e017e0c19b324245fef049dff2a7a2e49b9a391667ca9c4f667c4f683b1f49",
    },
)
