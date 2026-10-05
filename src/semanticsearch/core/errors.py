"""Engine exception hierarchy. Callers can catch SemanticSearchError for anything we raise."""


class SemanticSearchError(Exception):
    pass


class PackError(SemanticSearchError):
    """A domain pack is missing, malformed, or references files that don't exist."""


class ParseError(SemanticSearchError):
    """A source document could not be parsed."""


class CapabilityError(SemanticSearchError):
    """A component was asked for something its backend cannot do."""
