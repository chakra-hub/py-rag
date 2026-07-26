import re

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE),
    re.compile(r"forget\s+your\s+instructions", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"developer\s+message", re.IGNORECASE),
]

SQL_INJECTION_PATTERNS = [
    re.compile(r"\bunion\s+select\b", re.IGNORECASE),
    re.compile(r"\bdrop\s+table\b", re.IGNORECASE),
    re.compile(r"\binsert\s+into\b", re.IGNORECASE),
    re.compile(r"--"),
]

XSS_PATTERNS = [
    re.compile(r"<script", re.IGNORECASE),
    re.compile(r"javascript:", re.IGNORECASE),
    re.compile(r"onerror=", re.IGNORECASE),
]

COMMAND_PATTERNS = [
    re.compile(r"rm\s+-rf", re.IGNORECASE),
    re.compile(r"sudo\s+", re.IGNORECASE),
    re.compile(r"chmod\s+777", re.IGNORECASE),
]