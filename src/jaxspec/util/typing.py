from typing import Any

import numpyro.distributions as dist

from jax import numpy as jnp
from jax.typing import ArrayLike
from pydantic import BaseModel, field_validator

PriorDictType = dict[str, dict[str, dist.Distribution | ArrayLike]]


def is_flat_dict(input_data: dict[str, Any]) -> bool:
    """
    Check if the input data is a flat dictionary with string keys and non-dictionary values.
    """
    return all(isinstance(k, str) and not isinstance(v, dict) for k, v in input_data.items())


class PriorDictModel(BaseModel):
    """
    Pydantic model for a nested dictionary of NumPyro distributions or JAX arrays.
    The top level keys are strings, and the values are dictionaries with string keys and values that are either
    NumPyro distributions or JAX arrays (or convertible to JAX arrays).
    """

    nested_dict: PriorDictType

    class Config:  # noqa D106
        arbitrary_types_allowed = True

    @classmethod
    def from_dict(cls, input_prior: dict[str, Any]):
        if is_flat_dict(input_prior):
            nested_dict = {}

            for key, obj in input_prior.items():
                component, component_number, *parameter = key.split("_")

                sub_dict = nested_dict.get(f"{component}_{component_number}", {})
                sub_dict["_".join(parameter)] = obj

                nested_dict[f"{component}_{component_number}"] = sub_dict

            return cls(nested_dict=nested_dict)

        return cls(nested_dict=input_prior)

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
                            f'The value for key "{inner_key}" in {key} be a NumPyro '
                            f"distribution or castable to JAX array. Error: {e}"
                        )
        return value
