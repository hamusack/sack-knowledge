class LocalCommandError(ValueError):
    """A fixed diagnostic safe to display without inputs or local paths."""


class ModelIntegrityError(LocalCommandError):
    def __init__(self):
        super().__init__("Embedding asset SHA-256 mismatch. Restore the verified model files and retry.")
