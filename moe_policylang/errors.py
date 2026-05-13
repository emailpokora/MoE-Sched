"""Custom exceptions for MoE-Sched."""


class DSLError(Exception):
    """Raised when a DSL definition is malformed."""

    def __init__(self, message: str, line: int | None = None):
        self.line = line
        if line is not None:
            message = f"line {line}: {message}"
        super().__init__(message)


class ValidationError(Exception):
    """Raised when a PolicyIR fails validation."""

    def __init__(self, violations: list[str]):
        self.violations = violations
        msg = "Policy validation failed:\n" + "\n".join(f"  - {v}" for v in violations)
        super().__init__(msg)
