import torch

# Registry for frequency-domain JND effect base models
_FREQUENCY_EFFECT_REGISTRIES = {
    'CSF': {},  # contrast sensitivity function
    'LA': {},   # luminance adaptation
    'CM': {},   # contrast masking
    'TM': {},   # temporal masking
    'SA': {}    # saliency adaptation
}


class FrequencyEffectBase(torch.nn.Module):
    """
    Base class for frequency-domain JND effects.
    All specific frequency-domain JND models should inherit from this class
    and implement the predict() method.
    """
    def __init__(self, **params):
        super().__init__()
        self.params = params

    def predict(self, yuv: torch.Tensor) -> torch.Tensor:
        """
        Compute the frequency-domain JND map.

        Args:
            yuv: Input YUV tensor of shape (C, H, W).

        Returns:
            torch.Tensor: Single-channel JND map of shape (H, W).
        """
        raise NotImplementedError


def register_frequency_effect(effect_type: str, model_name: str = None):
    """
    Decorator to register a frequency-domain JND model class into
    _FREQUENCY_EFFECT_REGISTRIES.
    """
    def _decorator(cls):
        name = model_name or cls.__name__
        if effect_type not in _FREQUENCY_EFFECT_REGISTRIES:
            raise KeyError(f"Unknown effect type '{effect_type}'")
        _FREQUENCY_EFFECT_REGISTRIES[effect_type][name] = cls
        return cls
    return _decorator


def get_frequency_effect_registry():
    """Expose the frequency-domain JND effect registry."""
    return _FREQUENCY_EFFECT_REGISTRIES


# Registry for spatial-domain JND effect base models
_SPATIAL_EFFECT_REGISTRIES = {
    'LA': {},  # luminance adaptation
    'CM': {},  # contrast masking
    'PM': {},  # pattern masking
    'TM': {}   # temporal masking
}


class SpatialEffectBase(torch.nn.Module):
    """
    Base class for spatial-domain JND effects.
    All specific spatial-domain JND models should inherit from this class
    and implement the predict() method.
    """
    def __init__(self, **params):
        super().__init__()
        self.params = params

    def predict(self, yuv: torch.Tensor) -> torch.Tensor:
        """
        Compute the spatial-domain JND map.

        Args:
            yuv: Input YUV tensor of shape (C, H, W).

        Returns:
            torch.Tensor: Single-channel JND map of shape (H, W).
        """
        raise NotImplementedError


def register_spatial_effect(effect_type: str, model_name: str = None):
    """
    Decorator to register a spatial-domain JND model class into
    _SPATIAL_EFFECT_REGISTRIES.
    """
    def _decorator(cls):
        name = model_name or cls.__name__
        if effect_type not in _SPATIAL_EFFECT_REGISTRIES:
            raise KeyError(f"Unknown effect type '{effect_type}'")
        _SPATIAL_EFFECT_REGISTRIES[effect_type][name] = cls
        return cls
    return _decorator


def get_spatial_effect_registry():
    """Expose the spatial-domain JND effect registry."""
    return _SPATIAL_EFFECT_REGISTRIES