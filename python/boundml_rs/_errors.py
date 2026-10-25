"""The single exception type raised by BoundML."""


class BoundMLError(Exception):
    """A BoundML error. ``code`` is the stable ``BML-xxxx`` code (see docs/errors.md)."""

    def __init__(self, message: str, code: str = "BML-E000"):
        super().__init__(message)
        self.code = code
