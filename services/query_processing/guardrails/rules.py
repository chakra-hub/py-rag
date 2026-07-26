from abc import ABC, abstractmethod
from ..models import GuardrailResult
from ..models import GuardrailResult
from .patterns import PROMPT_INJECTION_PATTERNS, SQL_INJECTION_PATTERNS


class GuardrailRule(ABC):

    @abstractmethod
    def check(self, query: str) -> GuardrailResult:
        pass


class PromptInjectionRule(GuardrailRule):
    def check(self, query: str) -> GuardrailResult:
        for pattern in PROMPT_INJECTION_PATTERNS:
            if pattern.search(query):
                return GuardrailResult(
                    blocked=True,
                    reason="Prompt injection detected."
                )
        return GuardrailResult(blocked=False)
    
class SQLInjectionRule(GuardrailRule):
    def check(self, query: str):
        for pattern in SQL_INJECTION_PATTERNS:
            if pattern.search(query):
                return GuardrailResult(
                    blocked=True,
                    reason="SQL injection detected."
                )
        return GuardrailResult(blocked=False)
    
