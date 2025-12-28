# RAG

Sumber dataset : 
-  [**FC Barcelona Annual Report 2024/25**](https://www.fcbarcelona.com/fcbarcelona/document/2025/10/20/a50c257f-0ea4-4a92-b6b0-cacf189801fc/MEM_2024_25_ENG_web.pdf)
- [**Indofood Financial Results Summary 2025 (9 Months)**](https://www.indofood.com/uploads/statement/ISM_Billingual_30_September_2025_8%20Pages.pdf)
- [**Indofood Annual Report 2024**](https://www.indofood.com/uploads/annual/AR%20ISM%202024_web.pdf)



Flow : 
```
PDF
 → extract text (konsep awal di RAG.ipynb)
 → clean (sebelum TextSplitter)
 → chunk (TextSplitter di VectorStores_and_Retrievals.ipynb)
 → embedding (VectorStores_and_Retrievals.ipynb)
 → vector store FAISS (VectorStores_and_Retrievals.ipynb)
 → query (RAG.ipynb)
 → retrieve (VectorStores_and_Retrievals.ipynb)
 → prompt LLM (RAG.ipynb)
 → answer
```