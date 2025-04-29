### **Retrieval-Augmented Generation (RAG) Pipeline for Indian Legal AI Assistant**

The system is designed to provide legally grounded, section-specific assistance by combining semantic search with Large Language Model (LLM) capabilities. The architecture is both modular and efficient, enabling expert legal responses strictly based on authoritative Indian law sources.

---

#### **Pipeline Components**

1. **Corpus Preparation**  
   - **Input**: 8 major Indian legal acts in `.json` format:
     - Indian Penal Code (IPC), 1860  
     - Code of Criminal Procedure (CrPC), 1973  
     - Civil Procedure Code (CPC), 1908  
     - Indian Evidence Act (IEA), 1872  
     - Negotiable Instruments Act (NIA), 1881  
     - Hindu Marriage Act (HMA), 1955  
     - Indian Divorce Act (IDA), 1869  
     - The Motor Vehicles Act (MVA), 1988  
   - Each act is preprocessed to extract **Section**, **Title**, and **Description** fields.

2. **Semantic Embedding**
   - **Model Used**: `all-MiniLM-L6-v2` from SentenceTransformers.
   - Legal section descriptions are embedded into high-dimensional vectors for semantic similarity matching.

3. **FAISS Vector Indexing**
   - The embedded vectors are indexed using **Facebook AI Similarity Search (FAISS)** for fast retrieval.
   - The corresponding metadata (section text and titles) are stored in a parallel `.pkl` file.

4. **User Query Handling**
   - A user enters a free-form legal query via CLI.
   - The query is embedded and searched against the FAISS index to retrieve the top relevant legal sections.

5. **Prompt Construction**
   - Retrieved legal sections are compiled into a strict-format prompt.
   - The prompt explicitly instructs the LLM to only respond based on retrieved legal data—ensuring factual, non-speculative, citation-based outputs.

6. **LLM Integration**
   - **Model Used**: `deepseek-r1-distill-llama-70b` served via the Groq API.
   - The LLM processes the prompt and generates a context-aware, legally accurate response.

7. **Response Logging**
   - Each interaction (user query, matched sections, and response) is saved to a timestamped `.txt` file for traceability.
   - Saved in the `legal_outputs/` directory.

---

#### **Key Features**

- **Explainable**: Every answer is backed by actual law sections.
- **Restricted Generation**: The LLM cannot fabricate legal information—responses are grounded strictly in retrieved context.
- **Extensible**: New acts can be easily integrated into the system.
- **Efficient**: FAISS enables sub-second retrieval over thousands of entries.

---

This RAG pipeline ensures that users receive high-quality, section-cited legal assistance—bridging modern AI with traditional legal codification.

---
