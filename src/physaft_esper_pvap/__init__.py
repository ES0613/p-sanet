"""Public API for Esper-derived pure-component PC-SAFT properties."""

from .pipeline import PredictionResult, predict, predict_range

__version__ = "1.0.0"
__framework_name__ = "PhySAFT Esper"

__all__ = ["PredictionResult", "predict", "predict_range", "__framework_name__"]
