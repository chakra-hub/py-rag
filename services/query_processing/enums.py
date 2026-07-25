from enum import Enum

class QueryIntent(str, Enum):
    QUESTION="question"
    SEARCH="search"
    GREETING="greeting"
    FOLLOW_UP="greeting"
    SUMMARIZATION="summarization"
    COMPARISON="comparison"
    OUT_OF_SCOPE="out_of_scope"

class GuardrailType(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    ABUSE = "abuse"