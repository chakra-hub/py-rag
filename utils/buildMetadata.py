def _build_metadata(
    chunk,
    document_id: str,
    database_name: str,
    collection_name: str,
    version: str,
    chunk_index: int,
):
    """
    Build Chroma-compatible metadata from a Docling chunk.
    """

    page_numbers = sorted(
        {
            prov.page_no
            for item in chunk.meta.doc_items
            for prov in item.prov
        }
    )

    return {
        "doc_id": document_id,
        "doc_name": chunk.meta.origin.filename,
        "database_name": database_name,
        "collection_name": collection_name,
        "version": version,
        "chunk_index": chunk_index,

        # Store as strings because Chroma metadata doesn't support lists
        "page_numbers": ",".join(map(str, page_numbers)),

        "headings": (
            " | ".join(chunk.meta.headings)
            if chunk.meta.headings
            else ""
        ),

        "captions": (
            " | ".join(chunk.meta.captions)
            if chunk.meta.captions
            else ""
        ),

        "mime_type": chunk.meta.origin.mimetype,
        "source_file": chunk.meta.origin.filename,
    }