Introduction
========================

.. figure:: /Documentation/images/intro.jpg
   :width: 700
   :align: center
   :alt: Image explaining RAG introduction

--------------------------------------------------------------

Retrieval-Augmented Generation (RAG) is a powerful technique combining retrieval systems and generative models to produce more accurate and context-aware outputs.

1. Highlights
=============

- **Multilingual Support**: This application supports French, Arabic, and English.

- **Interactive Interfaces**: Built using frameworks like Streamlit.

2. General Pipeline
===================

**Input Query**
---------------

A user provides a natural language query (e.g., "What are the applications of quantum computing?").

**Retrieval System**
--------------------

- Query Encoder: The input query is encoded into a dense vector using a query encoder (embedder).
- Knowledge Base: A large dataset or knowledge base or corpus serving as a context is embedded into dense vectors and then stored in a database (such as chroma database)
- Similarity Search: The query vector is compared with document vectors using a similarity metric (e.g., cosine similarity).

**Natural Language Generation**
-------------------

A generative language model (e.g., GPT, llama, or BART) takes the combined input and generates a coherent, contextually relevant response.

**Output**
-------------------

The final response is returned to the user, enhanced by the external knowledge retrieved from the knowledge base.