import re
import unicodedata


class QueryNormalizer:
    """
    Normalizes user queries without changing their meaning.
    """

    INVISIBLE_CHARACTERS = [
        "\u200b",  # Zero Width Space
        "\u200c",  # Zero Width Non-Joiner
        "\u200d",  # Zero Width Joiner
        "\ufeff",  # Byte Order Mark (BOM)
    ]

    def normalize(self, query: str) -> str:
        """
        Normalize a query for downstream retrieval.

        Operations:
        1. Unicode normalization
        2. Remove invisible characters
        3. Normalize whitespace
        4. Strip leading/trailing spaces
        """

        # Unicode normalization
        query = unicodedata.normalize("NFKC", query)

        # Remove invisible unicode characters
        for char in self.INVISIBLE_CHARACTERS:
            query = query.replace(char, "")

        # Replace multiple newlines/tabs with space
        query = re.sub(r"\s+", " ", query)

        # Remove leading/trailing whitespace
        query = query.strip()

        return query
    