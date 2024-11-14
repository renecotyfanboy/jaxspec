from watermark import watermark


def debug_info():
    """
    Script which displays useful information about the user system and environment
    """

    # Cimer CamilleTheBest pour l'idée
    print(watermark())
    print(watermark(packages="jaxspec,jax,jaxlib,numpyro,flax,numpy,scipy"))
