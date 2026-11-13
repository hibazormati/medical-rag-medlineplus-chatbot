
# Medical RAG Chatbot
### Evidence-Grounded Health Education with Retrieval-Augmented Generation

<p align="center">
  <b>Hiba Zormati</b> · Statistics & Data Science Engineering · ESSAI Tunis
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LLM-Qwen2.5-6366F1" />
  <img src="https://img.shields.io/badge/RAG-FAISS-16A085" />
  <img src="https://img.shields.io/badge/NLP-Sentence_Transformers-orange" />
  <img src="https://img.shields.io/badge/Platform-Google_Colab-F9AB00?logo=googlecolab&logoColor=white" />
</p>

---

## Overview

Can a Large Language Model provide more trustworthy medical information when its answers are grounded in retrieved evidence rather than relying exclusively on its internal knowledge?

This project explores this question through the development and evaluation of a **Retrieval-Augmented Generation (RAG) system** for general health education.

The chatbot retrieves relevant information from **MedlinePlus**, the health-information service of the U.S. National Library of Medicine (NLM), and uses Qwen2.5 to generate answers grounded in the retrieved passages.

Beyond implementation, the project investigates retrieval quality, answer grounding, hallucination, uncertainty handling and the limitations of similarity-based refusal mechanisms.

> **Research focus:** Does retrieval improve answer grounding compared with the same language model operating without retrieval?

## System Architecture

The implementation consists of two complementary pipelines.

### 1. Offline indexing pipeline

```text
MedlinePlus Web Service
          |
          v
   XML Extraction
          |
          v
  HTML Cleaning & Parsing
          |
          v
 Document Chunking
 (sentence overlap)
          |
          v
 MiniLM Embeddings
   (384 dimensions)
          |
          v
    FAISS Index
```

### 2. Online question-answering pipeline

```text
    User Question
          |
          v
  Emergency Keyword Guard
          |
          v
  Semantic Retrieval
          |
          v
     FAISS Search
          |
          v
  MMR Re-ranking
          |
          v
 Similarity Threshold
          |
     +----+----+
     |         |
 Sufficient  Insufficient
 Evidence     Evidence
     |         |
     v         v
 Qwen2.5     Refusal
     |
     v
 Grounded Answer
     |
     v
 Source References
```

## Technical Implementation

| Component | Technology | Purpose |
|---|---|---|
| Data source | MedlinePlus API | Medical information retrieval |
| Data processing | Python, XML, Regex | Document extraction and cleaning |
| Chunking | Custom Python pipeline | Passage segmentation and overlap |
| Embeddings | all-MiniLM-L6-v2 | Semantic representations |
| Vector database | FAISS | Similarity search |
| Retrieval | Top-k + MMR | Relevance and diversity |
| Language model | Qwen2.5-3B-Instruct | Answer generation |
| Deep learning | PyTorch, Transformers | Model inference |
| Environment | Google Colab | GPU-based experimentation |

### Retrieval and Generation

**Semantic retrieval**

Text passages and user questions are encoded into 384-dimensional vectors using Sentence Transformers. L2 normalization allows cosine similarity to be computed through FAISS inner-product search.

**MMR re-ranking**

Maximal Marginal Relevance balances query relevance and diversity, reducing the selection of highly redundant passages.

**Context-grounded generation**

Qwen2.5-3B-Instruct receives the retrieved passages and a carefully designed system prompt. It is instructed to provide cautious educational answers, reference supporting sources and acknowledge insufficient evidence.

**Safety mechanisms**

- Similarity-based refusal when retrieved evidence falls below a calibrated threshold.
- Emergency keyword detection with a predefined safety message.
- Explicit instructions against diagnosis and unsupported medical claims.

These mechanisms reduce certain risks but do not guarantee medical correctness.

## Experimental Evaluation

The system was evaluated using eight questions covering different scenarios:

- General medical information.
- Disease prevention.
- Multiple-passage retrieval.
- Questions about subjects missing from the corpus.
- Personalized treatment requests.
- False premises.
- Potentially unsafe medical questions.
- Out-of-domain questions.

### Systems Compared

| System | Description |
|---|---|
| B0 | Same LLM without retrieval and with a generic prompt |
| B0-m | Same LLM without retrieval, with matched safety instructions |
| RAG | Full retrieval-augmented generation pipeline |

### Evaluation Methods

- Precision@k and Mean Reciprocal Rank (MRR).
- Claim-level and span-level evidence checks.
- Similarity threshold analysis.
- Qualitative evaluation of correctness, grounding, completeness and uncertainty handling.
- Latency and resource measurements.

### Selected Experimental Results

| Measurement | Observed result |
|---|---:|
| MedlinePlus documents | 24 |
| Health topics | 12 |
| Indexed chunks | 122 |
| Embedding dimensions | 384 |
| Retrieval threshold | 0.40 |
| Median RAG latency | 4.59 seconds |
| Maximum allocated GPU memory | 6.47 GB |

*Measurements correspond to the recorded October 13, 2026 Google Colab T4 run.*

### Key Findings

**1. Improved traceability**

Retrieved answers included identifiable source documents, making the information easier to inspect.

**2. Reduced unsupported specificity**

Compared with the generic no-retrieval baseline, the RAG system avoided several unsupported specific details, including certain drug names and numerical recommendations.

**3. Grounding is not a guarantee of faithfulness**

One experiment exposed an important failure: the model attributed a claim to a retrieved passage that did not actually contain that information.

**4. Similarity thresholds have limitations**

The 0.40 threshold separated the calibration examples but struggled with questions falling between clearly relevant and clearly irrelevant similarity scores.

These findings demonstrate why retrieval quality, generation behavior and evidence verification must be evaluated separately.

## Limitations

This project is an experimental prototype, not a clinically validated medical system.

Important limitations include:

- Relatively small and imbalanced corpus.
- Limited evaluation set of eight questions.
- Single-annotator relevance judgments.
- Similarity threshold calibrated on a small set rather than independently validated.
- Potential hallucinations and incorrect source attribution.
- Keyword-based emergency detection with possible false positives and false negatives.
- No independent clinical validation.

### Future Improvements

Potential research directions include:

- Hybrid BM25 and dense retrieval.
- Cross-encoder re-ranking.
- Larger and more balanced medical corpora.
- More robust answer/refusal calibration.
- Automated verification of claims against cited passages.
- Multi-annotator evaluation and expert review.
- Improved multilingual and context-aware safety mechanisms.

## Running the Project

The implementation is provided as a Jupyter notebook designed for Google Colab.

1. Open the notebook from this repository.
2. Upload it to Google Colab.
3. Select a GPU runtime if available.
4. Execute the installation and setup cells.
5. Run the data collection, preprocessing and indexing pipeline.
6. Load the language model and execute the evaluation cells.

The notebook includes dependency-version recording and a corpus snapshot mechanism to support reproducibility.

## Data Source

The project uses MedlinePlus health-topic information provided by the U.S. National Library of Medicine.

- [MedlinePlus](https://medlineplus.gov/)
- [MedlinePlus Web Services](https://medlineplus.gov/about/developers/webservices/)
- [NLM Copyright Information](https://medlineplus.gov/copyright.html)

## References

1. Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
2. Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.
3. Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP.
4. Carbonell & Goldstein (1998). The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries.
5. Qwen Team (2024). Qwen2.5 Technical Report.

## Disclaimer

This project is intended exclusively for educational and research purposes.

It does not provide medical diagnosis, personalized treatment or professional medical advice. Its responses may be incomplete or incorrect. Always consult qualified healthcare professionals for medical decisions. In emergencies, contact local emergency services.

---

<p align="center">
  <b>Built by Hiba Zormati</b><br/>
  Statistics & Data Science Engineering · ESSAI Tunis<br/>
  <a href="https://www.linkedin.com/in/hiba-zormati-624b13319/">LinkedIn</a> ·
  <a href="https://github.com/hibazormati">GitHub</a>
</p>

