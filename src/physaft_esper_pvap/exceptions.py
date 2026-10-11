"""Package-specific exceptions."""


class PCSAFTMLPvapError(RuntimeError):
    """Base exception for public pipeline failures."""


class InputValidationError(ValueError, PCSAFTMLPvapError):
    """Raised when SMILES or temperature inputs are invalid."""


class GeometryError(PCSAFTMLPvapError):
    """Raised when the required conformer calculation is incomplete."""


class ModelAssetError(PCSAFTMLPvapError):
    """Raised when model assets are absent or fail integrity checks."""


class StrictPredictionError(PCSAFTMLPvapError):
    """Raised when strict mode encounters any failed prediction point."""
