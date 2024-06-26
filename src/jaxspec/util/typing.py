from typing import Any

import numpyro.distributions as dist

from jax import numpy as jnp
from jax.typing import ArrayLike
from pydantic import BaseModel, field_validator

PriorDictType = dict[str, dict[str, dist.Distribution | ArrayLike]]


class PriorDictModel(BaseModel):
    """
    Pydantic model for a nested dictionary of NumPyro distributions or JAX arrays.
    The top level keys are strings, and the values are dictionaries with string keys and values that are either
    NumPyro distributions or JAX arrays (or convertible to JAX arrays).
    """

    nested_dict: PriorDictType

    class Config:  # noqa D106
        arbitrary_types_allowed = True

    @field_validator("nested_dict", mode="before")
    def check_and_cast_nested_dict(cls, value: dict[str, Any]):
        if not isinstance(value, dict):
            raise ValueError("The top level must be a dictionary")

        for key, inner_dict in value.items():
            if not isinstance(inner_dict, dict):
                raise ValueError(f'The value for key "{key}" must be a dictionary')

            for inner_key, obj in inner_dict.items():
                if not isinstance(obj, dist.Distribution):
                    try:
                        # Attempt to cast to JAX array
                        value[key][inner_key] = jnp.array(obj, dtype=float)
                    except Exception as e:
                        raise ValueError(
                            f'The value for key "{inner_key}" in inner dictionary must '
                            f"be a NumPyro distribution or castable to JAX array. Error: {e}"
                        )
        return value
