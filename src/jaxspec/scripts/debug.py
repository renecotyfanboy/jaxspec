"""Console script printing the information needed to triage a bug report."""

import importlib.metadata
import platform
import sys

# Reported in this order: jaxspec first, then the stack it sits on.
_PACKAGES = (
    "jaxspec",
    "jax",
    "jaxlib",
    "numpyro",
    "flax",
    "numpy",
    "scipy",
    "astropy",
    "arviz",
)


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _backend() -> str:
    """Report the JAX backend, the usual culprit behind platform-specific bugs."""
    try:
        import jax

        return f"{jax.default_backend()} — {len(jax.devices())} device(s): {jax.devices()}"
    except Exception as exc:  # pragma: no cover - only on a broken local install
        return f"unavailable ({type(exc).__name__}: {exc})"


def debug_info():
    """Display useful information about the user system and environment."""

    width = max(len(package) for package in _PACKAGES) + 1

    print("System")
    print(f"  {'python':<{width}}: {platform.python_version()} ({sys.executable})")
    print(f"  {'platform':<{width}}: {platform.platform()}")
    print(f"  {'machine':<{width}}: {platform.machine()}")

    print("\nPackages")
    for package in _PACKAGES:
        print(f"  {package:<{width}}: {_version(package)}")

    print("\nJAX")
    print(f"  {'backend':<{width}}: {_backend()}")


if __name__ == "__main__":
    debug_info()
