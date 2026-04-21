"""User-facing domain errors raised by the service layer."""
from __future__ import annotations


class DomainError(Exception):
    """Base class for errors the GUI is allowed to display to the user."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


class ModelIOError(DomainError):
    """Failure loading or saving a model file."""


class CompileError(DomainError):
    """Failure during junction-tree compile / update_beliefs."""


class QueryError(DomainError):
    """Failure during a single- or batch-query operation."""


class EvidenceError(DomainError):
    """Invalid evidence supplied by the user."""
