

from models.collection_info import CollectionInfo


class CollectionRegistry:
    """
    Responsible for managing collection versions.

    This is the only class that knows how collections are versioned.
    """

    def get_active_collection(self) -> CollectionInfo:
        """
        Returns the latest available collection.

        Example:
            hr_v7
        """
        raise NotImplementedError()

    def create_next_collection(self) -> CollectionInfo:
        """
        Creates the next collection version.

        Example:
            hr_v8
        """
        raise NotImplementedError()