"""Document Question Answering System (RAG) — Week 7.

A config-driven Retrieval-Augmented Generation pipeline:
ingest -> chunk -> embed -> vector store -> retrieve -> generate (Gemini),
plus a rigorous, ground-truth evaluation harness.

Importing this package is cheap; heavy dependencies (torch, chromadb) are
only pulled in when you import the submodule that needs them.
"""

__version__ = "0.1.0"
