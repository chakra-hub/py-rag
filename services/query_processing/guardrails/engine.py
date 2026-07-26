from services.query_processing.models import GuardrailResult
from .rules import (
    PromptInjectionRule,
    SQLInjectionRule,
)


class GuardrailEngine:

    def __init__(self):
        self.rules = [
            PromptInjectionRule(),
            SQLInjectionRule(),
        ]

    def check(self, query: str):
        for rule in self.rules:
            result = rule.check(query)
            if result.blocked:
                return result
        return GuardrailResult(blocked=False)
