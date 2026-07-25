import re

from .models import ValidationResult


class InputValidator:

    MAX_QUERY_LENGTH = 1000

    MAX_REPEATED_CHAR = 15

    def validate(self, query: str) -> ValidationResult:

        errors = []

        if query is None:
            errors.append("Query cannot be None.")

        elif len(query.strip()) == 0:
            errors.append("Query cannot be empty.")

        if len(query) > self.MAX_QUERY_LENGTH:
            errors.append(
                f"Query exceeds maximum length ({self.MAX_QUERY_LENGTH})."
            )

        repeated_pattern = r"(.)\1{" + str(self.MAX_REPEATED_CHAR) + ",}"

        if re.search(repeated_pattern, query):
            errors.append("Repeated characters detected.")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
        )
    