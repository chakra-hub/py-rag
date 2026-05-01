from docling.document_converter import DocumentConverter

def extractDocsFromRequest(document):
    converter = DocumentConverter()
    return converter.convert(document)