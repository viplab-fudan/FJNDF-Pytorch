import torch
# frequency-domain JND effect base model
_FREQUENCY_EFFECT_REGISTRIES = {
    'CSF': {}, # contrast sensitive factor
    'LA': {},  # luminance adaptation
    'CM': {},  # contrast masking
    'TM': {},  # temporal masking
    'SA': {}   # saliency adaptation
}

class FrequencyEffectBase(torch.nn.Module):
    """
    Frequency-domain JND effect base class, all specific models should inherit and implement predict()
    """
    def __init__(self, **params):
        super().__init__()
        self.params = params

    def predict(self, yuv: torch.Tensor) -> torch.Tensor:
        """
        Input: yuv tensor (C,H,W)
        OUtput: Single channel JND map tensor (H,W)
        """
        raise NotImplementedError

def register_frequency_effect(effect_type: str, model_name: str = None):
    """Decorator: Mark a subclass and automatically write to _FREQUENCY_EFFECT_REGISTRIES."""
    def _decorator(cls):
        name = model_name or cls.__name__
        if effect_type not in _FREQUENCY_EFFECT_REGISTRIES:
            raise KeyError(f"Unknown effect type '{effect_type}'")
        _FREQUENCY_EFFECT_REGISTRIES[effect_type][name] = cls
        return cls
    return _decorator

# Expose registry externally
def get_frequency_effect_registry():
    return _FREQUENCY_EFFECT_REGISTRIES