"""Custom exceptions for Folio."""


class FolioError(Exception):
    """Base exception for all Folio errors."""


class ProjectNotFoundError(FolioError):
    """Raised when a project doesn't exist."""


class ProjectExistsError(FolioError):
    """Raised when creating a project that already exists."""


class InvalidSchemaError(FolioError):
    """Raised when project schema is malformed."""


class InvalidInputError(FolioError):
    """Raised when input values are invalid or outside bounds."""


class InvalidOutputError(FolioError):
    """Raised when output values are invalid."""


class InsufficientDataError(FolioError):
    """Raised when there's not enough data for an operation."""


class RecommenderError(FolioError):
    """Raised when recommender fails to suggest."""


class SurrogateError(FolioError):
    """Raised when surrogate model fails."""
