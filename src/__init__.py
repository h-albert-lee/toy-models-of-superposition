"""Public module exports for the toolkit."""

# Conditional imports to avoid torch dependency when only using data loaders
try:
    from .feature_extractor import VectorExtractor
    from .vlm_wrapper import VLM_Wrapper
    _torch_available = True
except ImportError:
    VectorExtractor = None
    VLM_Wrapper = None
    _torch_available = False

from .data_utils import load_pairs, load_cases

__all__ = [
    "VectorExtractor",
    "VLM_Wrapper", 
    "load_pairs",
    "load_cases",
]

# Only export torch-dependent classes if available
if not _torch_available:
    __all__ = [name for name in __all__ if name not in ["VectorExtractor", "VLM_Wrapper"]]
