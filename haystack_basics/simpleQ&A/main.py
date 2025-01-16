# A simple indexing_pipeline example using Haystack
# This is a question and answering service
from pathlib import Path
from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.converters import PyPDFToDocument
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import (
    InMemoryEmbeddingRetriever,
)
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.readers import ExtractiveReader

pdf_path = "./data/CFS.pdf"
document_store = InMemoryDocumentStore()
converter = PyPDFToDocument()
docs = converter.run(sources=[Path("./data/CFS.pdf")])

document_store = InMemoryDocumentStore()

embedder = SentenceTransformersDocumentEmbedder(
    model="intfloat/e5-large-v2", prefix="passage"
)
embedder.warm_up()
converter = PyPDFToDocument()
splitter = DocumentSplitter()
embedder = SentenceTransformersDocumentEmbedder(
    model="intfloat/e5-large-v2", prefix="passage"
)
writer = DocumentWriter(document_store=document_store)

# Create and connect pipeline
indexing_pipeline = Pipeline()
indexing_pipeline.add_component("converter", converter)
indexing_pipeline.add_component("splitter", splitter)
indexing_pipeline.add_component("embedder", embedder)
indexing_pipeline.add_component("writer", writer)

# Connect components in sequence
indexing_pipeline.connect("converter", "splitter")
indexing_pipeline.connect("splitter", "embedder")
indexing_pipeline.connect("embedder", "writer")

indexing_pipeline.run({"converter": {"sources": ["data/CFS.pdf"]}})

# Query pipeline

query_embedder = SentenceTransformersTextEmbedder(
    model="intfloat/e5-large-v2", prefix="query"
)
retriever = InMemoryEmbeddingRetriever(document_store=document_store)
reader = ExtractiveReader()
reader.warm_up()

documents_search_pipeline = Pipeline()
documents_search_pipeline.add_component("query_embedder", query_embedder)
documents_search_pipeline.add_component("retriever", retriever)
documents_search_pipeline.add_component("reader", reader)
documents_search_pipeline.connect(
    "query_embedder.embedding", "retriever.query_embedding"
)
documents_search_pipeline.connect("retriever.documents", "reader.documents")


query = "What are the symptoms of chronic fatigue syndrome"
answer = documents_search_pipeline.run(
    data={
        "query_embedder": {"text": query},
        "retriever": {"top_k": 3},
        "reader": {"query": query, "top_k": 2},
    }
)
print(answer)
