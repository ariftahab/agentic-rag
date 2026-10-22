# Experiment Log

> Otomatik üretilir (`python -m experiments.report`). Production: **EXP-001**. Ayrıntılar: `experiments/results/EXP-*.json`.

**Dataset:** `eval/sorular_v1.json` vv1 (45 soru: 42 tool beklentili, 11 hesaplı). Bir chunk, doğru kaynak belgelerden birine aitse ve o belgede birebir geçen evidence cümlesini içeriyorsa relevant sayılır. Farklı setlerle koşulan deneyler birbiriyle karşılaştırılmaz.  
**Kural:** Her deney parent'ından tek bir parametre farklı. Hypothesis deneyden önce yazılır. p < 0,05 = anlamlı (McNemar / Wilcoxon, aynı sorular üzerinde).  
**Sabit parametreler:** `corpus=v1`, `device=cuda`, `reranker_dtype=fp16`, `fixed_answers=summer_v1`, `citation_style=madde`, `pdf_parser=pdfplumber`, `chunking=structural_v2`, `chunk_size=1500`, `chunk_overlap=150`, `embedding_model=bge-m3`, `normalize_embeddings=✓`, `retrieval=hybrid`, `vector_k=30`, `bm25_k=30`, `rrf_k=60`, `reranker_model=bge-reranker`, `reranker_max_length=512`, `rerank_candidates=30`, `final_k=5`, `llm_temperature=0.3`

## 1. Setup (hyperparameters)

Kalın hücre = parent'a göre değişen parametre.

| Exp | Parent | mod | corpus | device | parser | chunking | chunk_size | overlap | embedding | normalize | retrieval | reranker | final_k | prompt | llm | #chunks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP-001 | – | rag | v1 | cuda | pdfplumber | structural_v2 | 1500 | 150 | bge-m3 | ✓ | hybrid | bge-reranker | 5 | v1 | 3.1-flash-lite | – |
| EXP-002 | – | agent | v1 | cuda | pdfplumber | structural_v2 | 1500 | 150 | bge-m3 | ✓ | hybrid | bge-reranker | 5 | agent_v3 | 3.5-flash-lite | 685 |
| EXP-003 | EXP-002 | agent | v1 | cuda | pdfplumber | structural_v2 | 1500 | 150 | bge-m3 | ✓ | hybrid | bge-reranker | 5 | **agent_v4** | 3.5-flash-lite | 685 |

## 2. Results

| Exp | Değişiklik | Set | MRR@10 | Hit@1 | Hit@5 | Context hit | Context chars | Latency | Primary metric | Hyp. |
|---|---|---|---|---|---|---|---|---|---|---|
| EXP-001 | Başlangıç yapılandırması: campus-chatbot retrieval hattı, mühendislik corpus v1, tek prompt | | | | | | | | koşulmadı | |
| EXP-002 | Agent baseline: agent_v3 prompt, 6 tool, sorular_v1 | v1 | – | – | – | – | – | – | baseline |  |
| EXP-003 | agent_v4: oturumun transkript durumu prompt'ta | v1 | – | – | – | – | – | – | tool seçimi ↓ kötüleşti (p=1.00, anlamlı değil) | ✗ |

Hyp.: ✓ doğrulandı · ✗ doğrulanmadı · ~ yön doğru ama anlamlı değil

### Agent

Tool seçimi: beklenen tool'lar doğru argümanlarla çağrıldı, beklenmeyen tool yok. Tool adı: aynısı, argümanlara bakmadan. Hesap (cevap): elle hesaplanan AKTS/GNO cevapta tam olarak geçiyor. Hesap (tool): aynı sayılar deterministik modülün sonucuyla aynı (LLM'den bağımsız). Answer acc. anlamca ayrıca okunur.

| Exp | Set | Answer acc. | Tool seçimi | Tool adı | Hesap (cevap) | Hesap (tool) | Ort. tool çağrısı | Ort. tur | n | Tamamlandı |
|---|---|---|---|---|---|---|---|---|---|---|
| EXP-002 | v1 | 0.732 | 0.738 | 0.738 | 0.636 | 1.000 | 1.07 | 2.04 | 45 | ✓ |
| EXP-003 | v1 | 0.780 | 0.714 | 0.714 | 0.727 | 1.000 | 1.13 | 2.11 | 45 | ✓ |

## 5. Notes

- **EXP-001:** Başlangıç yapılandırması, koşu yok: campus-chatbot retrieval hattı + corpus v1 (mühendislik, 24 belge), tek prompt v1.
- **EXP-002:** Agent baseline: tool 0,74, hesap 0,64 (kod 1,00), cevap 0,73 (anlamca ~0,87); ana hata transkript yüklüyken kişisel sorularda analyze_transcript çağrılmaması.
- **EXP-003:** agent_v4: tool metriği 0,74 → 0,71 (zararsız ek çağrılar hata sayıldı), cevap 0,73 → 0,78 (anlamca ~0,87 → 0,93), hesap 0,64 → 0,73; transkript soruları düzeldi.
