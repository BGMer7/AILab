
# Retrieval-Augmented Generation for Large Language Models A Survey
## Introduction
## Overview of RAG
### Naive RAG
#### Indexing
#### Retrieval
#### Generation
#### Notable Drawbacks
##### Retrieval Challenges
##### Generation Difficulties
##### Augmentation Hurdles
### Advanced RAG
#### Pre-Retrieval process
##### query rewriting
##### query transformation
##### query expansion
#### Post-Retrieval process
### Modular RAG
#### New Modules
##### Search Module
##### Memory Module
##### Routing Module
##### Predict Module
##### Task Adapter Module
#### New Patterns
##### Iterative Retrieval
##### Recursive Retrieval
##### Adaptive Retrieval
##### Rewrite-Retrieve-Read
##### hypothetical document embeddings (HyDE)
##### Flexible and Adaptive Retrieval Augmented Generation (FLARE)
##### Self-RAG
## RAG vs Fine-tuning
## Retrieval
### Retrieval Source
#### Data Structure
##### Unstructured Data
##### Semi-structured Data
##### Structured Data
### Retrieval Granularity
### Indexing Optimization
#### Chunking Strategy
#### Metadata Attachments
#### Structural Index
### Query Optimization
#### Query Expansion
#### Query Transformation
#### Query Routing
### Embedding
#### Embedding Model
#### Mixed/Hybrid Retrieval
#### Embedding Model Fine-tuning
### Adapter
#### Lightweight Adapter
#### Pluggable Adapter
## Generation
### Context Curation
#### Reranking
#### Context Selection/Compression
### LLM Fine-tuning
## Augmentation Process
### Iterative Retrieval
#### ITER - RETGEN
### Recursive Retrieval
#### IRCoT
#### ToC
#### multi-hop retrieval
### Adaptive Retrieval
#### FLARE
#### Self-RAG
## Task And Evaluation
### Downstream Task
#### QA
##### Single/Multi hop QA
##### Domain-specific QA
### Evaluation Target
#### Historical Evaluation Approach
##### Task-Specific Metrics
##### Automated Evaluation Tools
#### Retrieval Quality
##### Hit Rate
##### Mean Reciprocal Rank (MRR)
##### Normalized Discounted Cumulative Gain (NDCG)
#### Generation Quality
##### Unlabeled Content
##### Labeled Content
#### Future Directions in Evaluation
### Evaluation Aspects
#### Quality Scores
##### Context Relevance
##### Answer Faithfulness
##### Answer Relevance
#### Required Abilities
##### Noise Robustness
##### Negative Rejection
##### Information Integration
##### Counterfactual Robustness
### Evaluation Benchmarks and Tools
#### Benchmarks
##### RGB
##### Recall
##### CRUD
#### Tools
##### RAGAS
##### ARES
## Discussion And Future Prospects
