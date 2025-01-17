from haystack import component
from langchain_experimental.text_splitter import SemanticChunker
from types import List, Protocol

"""
A component which wraps a semantic splitting function

"""


class DocumentType(Protocol):
    content: str
    metadata: List[dict]


class SenamticSpliter:

    def __init__(self, embedder, breakpoint_threshold_type="percentile"):
        self.embedder = embedder
        self.breakpoint_threshold_type = breakpoint_threshold_type

    @component.output_types(List[DocumentType])
    def run(self, doc: DocumentType) -> List[DocumentType]:
        splitter = SemanticChunker(
            self.embedder, breakpoint_threshold_type=self.breakpoint_threshold_type
        )

        return splitter.create_documents(texts=doc.content, metadatas=doc.metadatas)
