"""Likelihood model."""

from typing import Annotated, Literal, Union

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, PlainSerializer, PlainValidator

from careamics.models.lvae.noise_models import (
    GaussianMixtureNoiseModel,
    MultiChannelNoiseModel,
)
from careamics.utils.serializers import _array_to_json, _to_torch

NoiseModel = Union[GaussianMixtureNoiseModel, MultiChannelNoiseModel]

# TODO: this is a temporary solution to serialize and deserialize tensor fields
# in pydantic models. Specifically, the aim is to enable saving and loading configs
# with such tensors to/from JSON files during, resp., training and evaluation.
Tensor = Annotated[
    Union[np.ndarray, torch.Tensor],
    PlainSerializer(_array_to_json, return_type=str),
    PlainValidator(_to_torch),
]
"""Annotated tensor type, used to serialize arrays or tensors to JSON strings
and deserialize them back to tensors."""


class GaussianLikelihoodConfig(BaseModel):
    """Gaussian likelihood configuration."""

    model_config = ConfigDict(validate_assignment=True)

    predict_logvar: Literal["pixelwise"] | None = None
    """If `pixelwise`, log-variance is computed for each pixel, else log-variance
    is not computed."""

    logvar_lowerbound: Union[float, None] = None
    """The lowerbound value for log-variance."""


class NMLikelihoodConfig(BaseModel):
    """Noise model likelihood configuration.

    NOTE: we need to define the data mean and std here because the noise model
    is trained on not-normalized data. Hence, we need to unnormalize the model
    output to compute the noise model likelihood.
    """

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    # TODO remove and use as parameters to the likelihood functions?
    data_mean: Tensor = torch.zeros(1)
    """The mean of the data, used to unnormalize data for noise model evaluation.
    Shape is (target_ch,) (or (1, target_ch, [1], 1, 1))."""

    # TODO remove and use as parameters to the likelihood functions?
    data_std: Tensor = torch.ones(1)
    """The standard deviation of the data, used to unnormalize data for noise
    model evaluation. Shape is (target_ch,) (or (1, target_ch, [1], 1, 1))."""
