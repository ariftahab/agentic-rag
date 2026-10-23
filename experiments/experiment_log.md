# Experiment Log

> Otomatik üretilir (`python -m experiments.report`). Production: **EXP-007**. Ayrıntılar: `experiments/results/EXP-*.json`.

**Dataset:** `eval/sorular_v1.json` vv1 (45 soru: 42 tool beklentili, 11 hesaplı) · `eval/sorular_v2.json` vv2 (66 soru: 63 tool beklentili, 22 hesaplı) · `eval/sorular_v3.json` vv3 (66 soru: 63 tool beklentili, 22 hesaplı) · `eval/sorular_v4.json` vv4 (70 soru: 67 tool beklentili, 26 hesaplı). Bir chunk, doğru kaynak belgelerden birine aitse ve o belgede birebir geçen evidence cümlesini içeriyorsa relevant sayılır. Farklı setlerle koşulan deneyler birbiriyle karşılaştırılmaz.  
**Kural:** Her deney parent'ından tek bir parametre farklı. Hypothesis deneyden önce yazılır. p < 0,05 = anlamlı (McNemar / Wilcoxon, aynı sorular üzerinde).  
**Sabit parametreler:** `corpus=v1`, `device=cuda`, `reranker_dtype=fp16`, `fixed_answers=summer_v1`, `citation_style=madde`, `pdf_parser=pdfplumber`, `chunking=structural_v2`, `chunk_size=1500`, `chunk_overlap=150`, `embedding_model=bge-m3`, `normalize_embeddings=✓`, `retrieval=hybrid`, `vector_k=30`, `bm25_k=30`, `rrf_k=60`, `reranker_model=bge-reranker`, `reranker_max_length=512`, `rerank_candidates=30`, `final_k=5`, `llm_temperature=0.3`

## 1. Setup (hyperparameters)

Kalın hücre = parent'a göre değişen parametre.

| Exp | Parent | mod | corpus | device | parser | chunking | chunk_size | overlap | embedding | normalize | retrieval | reranker | final_k | prompt | llm | #chunks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP-001 | – | rag | v1 | cuda | pdfplumber | structural_v2 | 1500 | 150 | bge-m3 | ✓ | hybrid | bge-reranker | 5 | v1 | 3.1-flash-lite | – |
| EXP-002 | – | agent | v1 | cuda | pdfplumber | structural_v2 | 1500 | 150 | bge-m3 | ✓ | hybrid | bge-reranker | 5 | agent_v3 | 3.5-flash-lite | 685 |
| EXP-003 | EXP-002 | agent | v1 | cuda | pdfplumber | structural_v2 | 1500 | 150 | bge-m3 | ✓ | hybrid | bge-reranker | 5 | **agent_v4** | 3.5-flash-lite | 685 |
| EXP-004 | EXP-003 | agent | v1 | cuda | pdfplumber | structural_v2 | 1500 | 150 | bge-m3 | ✓ | hybrid | bge-reranker | 5 | **agent_v5** | 3.5-flash-lite | 685 |
| EXP-005 | EXP-004 | agent | v1 | cuda | pdfplumber | structural_v2 | 1500 | 150 | bge-m3 | ✓ | hybrid | bge-reranker | 5 | **agent_v7** | 3.5-flash-lite | 685 |
| EXP-006 | EXP-005 | agent | v1 | cuda | pdfplumber | structural_v2 | 1500 | 150 | bge-m3 | ✓ | hybrid | bge-reranker | 5 | agent_v7 | **3.1-flash-lite** | 685 |
| EXP-007 | EXP-001 | rag | v1 | cuda | pdfplumber | structural_v2 | 1500 | 150 | bge-m3 | ✓ | hybrid | bge-reranker | 5 | v1 | **3.5-flash-lite** | – |
| EXP-008 | EXP-005 | agent | v1 | cuda | pdfplumber | structural_v2 | 1500 | 150 | bge-m3 | ✓ | hybrid | bge-reranker | 5 | **agent_v10** | 3.5-flash-lite | 685 |
| EXP-009 | EXP-008 | agent | v1 | cuda | pdfplumber | structural_v2 | 1500 | 150 | bge-m3 | ✓ | hybrid | bge-reranker | 5 | **agent_v12** | 3.5-flash-lite | 685 |
| EXP-010 | EXP-009 | agent | v1 | cuda | pdfplumber | structural_v2 | 1500 | 150 | bge-m3 | ✓ | hybrid | bge-reranker | 5 | **agent_h1** | **claude-haiku-5-5** | 685 |
| EXP-011 | EXP-010 | agent | v1 | cuda | pdfplumber | structural_v2 | 1500 | 150 | bge-m3 | ✓ | hybrid | bge-reranker | 5 | **agent_h2** | claude-haiku-5-5 | 685 |

## 2. Results

| Exp | Değişiklik | Set | MRR@10 | Hit@1 | Hit@5 | Context hit | Context chars | Latency | Primary metric | Hyp. |
|---|---|---|---|---|---|---|---|---|---|---|
| EXP-001 | Başlangıç yapılandırması: campus-chatbot retrieval hattı, mühendislik corpus v1, tek prompt | | | | | | | | koşulmadı | |
| EXP-002 | Agent baseline: agent_v3 prompt, 6 tool, sorular_v1 | v1 | – | – | – | – | – | – | baseline |  |
| EXP-003 | agent_v4: oturumun transkript durumu prompt'ta | v1 | – | – | – | – | – | – | tool seçimi ↓ kötüleşti (p=1.00, anlamlı değil) | ✗ |
| EXP-004 | agent_v5: genel mezuniyet şartları için get_graduation_requirements | v1 | – | – | – | – | – | – | answer acc. ↑ iyileşti (p=0.38, anlamlı değil) | ~ |
| EXP-005 | agent_v7 + sorular_v2: anlık tool baseline'ı (gemini-3.5-flash-lite) | v2 | – | – | – | – | – | – | referans (yeni soru seti, karşılaştırma yok) | – |
| EXP-006 | Model karşılaştırması: gemini-3.1-flash-lite + agent_v7 (sorular_v2) | v2 | – | – | – | – | – | – | answer acc. ↓ kötüleşti (p<0.01, anlamlı) | ✓ |
| EXP-007 | Production modeli gemini-3.5-flash-lite (klasik RAG yolu) | | | | | | | | koşulmadı | |
| EXP-008 | agent_v10: devir + simulate_grades (sorular_v3) | v3 | – | – | – | – | – | – | referans (yeni soru seti, karşılaştırma yok) | – |
| EXP-009 | agent_v12: transkript raporuna ders geçmişi + raporda yoksa devir (sorular_v4) | v4 | – | – | – | – | – | – | referans (yeni soru seti, karşılaştırma yok) | – |
| EXP-010 | agent_h1: tek model Claude Haiku 5.5, Gemini yok (sorular_v4, branch exp/haiku-agent) | v4 | – | – | – | – | – | – | answer acc.: ölçülmedi | beklemede |
| EXP-011 | agent_h2: tek model Haiku + çok değerli hesaplar kodda; prompt caching (sorular_v4, branch exp/haiku-agent) | v4 | – | – | – | – | – | – | hesap = değişmedi (p=1.00, anlamlı değil) | ✗ |

Hyp.: ✓ doğrulandı · ✗ doğrulanmadı · ~ yön doğru ama anlamlı değil

### Agent

Tool seçimi: beklenen tool'lar doğru argümanlarla çağrıldı, beklenmeyen tool yok. Tool adı: aynısı, argümanlara bakmadan. Gerekli tool: beklenen tool'lar doğru argümanlarla çağrıldı, fazla çağrı serbest. Yan etkisiz: oturum durumunu değiştiren tool (teyit kaydı) beklenmeden çağrılmadı. Hesap (cevap): elle hesaplanan AKTS/GNO cevapta tam olarak geçiyor. Hesap (tool): aynı sayılar deterministik modülün sonucuyla aynı (LLM'den bağımsız). Answer acc. anlamca ayrıca okunur.

| Exp | Set | Answer acc. | Tool seçimi | Tool adı | Gerekli tool | Yan etkisiz | Hesap (cevap) | Hesap (tool) | Anlık tool isabeti | Gereksiz anlık tool | Haiku hesabı | Ort. tool çağrısı | Ort. tur | n | Tamamlandı |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP-002 | v1 | 0.732 | 0.738 | 0.738 | 0.809 | 0.976 | 0.636 | 1.000 | – | – | – | 1.07 | 2.04 | 45 | ✓ |
| EXP-003 | v1 | 0.780 | 0.714 | 0.714 | 0.905 | 1.000 | 0.727 | 1.000 | – | – | – | 1.13 | 2.11 | 45 | ✓ |
| EXP-004 | v1 | 0.854 | 0.691 | 0.691 | 0.833 | 1.000 | 0.727 | 1.000 | – | – | – | 1.22 | 2.13 | 45 | ✓ |
| EXP-005 | v2 | 0.875 | 0.603 | 0.603 | 0.809 | 1.000 | 0.500 | 1.000 | 0.879 | 0.038 | 0.750 | 1.29 | 2.24 | 66 | ✓ |
| EXP-006 | v2 | 0.679 | 0.682 | 0.682 | 0.762 | 1.000 | 0.455 | 1.000 | 0.894 | 0.000 | 0.571 | 0.98 | 2.03 | 66 | ✓ |
| EXP-008 | v3 | 0.857 | 0.698 | 0.698 | 0.841 | 1.000 | 0.773 | 1.000 | 0.894 | 0.053 | 1.000 | 1.17 | 2.02 | 66 | ✓ |
| EXP-009 | v4 | 0.923 | 0.635 | 0.635 | 0.905 | 1.000 | 0.846 | 1.000 | 0.939 | 0.069 | 0.875 | 1.36 | 2.05 | 66 | ✗ (kota) |
| EXP-010 | v4 | 1.000 | 0.687 | 0.687 | 0.940 | 1.000 | 0.962 | 1.000 | 0.929 | 0.032 | 1.000 | 1.54 | 2.06 | 70 | ✓ |
| EXP-011 | v4 | 1.000 | 0.642 | 0.642 | 0.925 | 1.000 | 0.962 | 1.000 | 0.914 | 0.065 | 1.000 | 1.53 | 2.07 | 70 | ✓ |

## 5. Notes

- **EXP-001:** Başlangıç yapılandırması, koşu yok: campus-chatbot retrieval hattı + corpus v1 (mühendislik, 24 belge), tek prompt v1.
- **EXP-002:** Agent baseline: tool 0,74, hesap 0,64 (kod 1,00), cevap 0,73 (anlamca ~0,87); ana hata transkript yüklüyken kişisel sorularda analyze_transcript çağrılmaması.
- **EXP-003:** agent_v4: tool metriği 0,74 → 0,71 (zararsız ek çağrılar hata sayıldı), cevap 0,73 → 0,78 (anlamca ~0,87 → 0,93), hesap 0,64 → 0,73; transkript soruları düzeldi.
- **EXP-004:** agent_v5: mezuniyet şartları sorusu tam ve kaynaklı (metrikler hata sayıyor, set eskidi); answer_acc 0,78 → 0,85; transkript yüklüyken "ADSD almam gerekiyor mu?" kötüleşti (yeni tool çaldı).
- **EXP-005:** agent_v7 baseline (sorular_v2): anlık tool program/şube sorularında doğru, ortalama zinciri hiç kurulmadı (katsayılar sandbox'ta yok), geçmiş not sorularında rapora güvenip yanlış; gereksiz tetiklenme 2/52.
- **EXP-006:** gemini-3.1-flash-lite: answer_acc 0,875 → 0,679 (p=0,003), 3 soru 503, latency ~4 kat; yanlış belge (lisansüstü not tablosu) ve tool sayılarını yeniden toplama hataları. Öneri: production 3.5-flash-lite.
- **EXP-007:** Production RagBot modeli 3.1-flash-lite → 3.5-flash-lite (EXP-006'ya dayanarak, koşulmadı).
- **EXP-008:** agent_v10: ortalama soruları 5/5 doğru, hesap_dogrulugu 0,50 → 0,77, devir cevapları 5/5; kalan: rapora güvenen yanlış tool seçimi (kaldığım dersler, dönem, ders sayısı), bir gereksiz devir, katsayı aktarım hatası.
- **EXP-009:** agent_v12 (yarım, kota 66/70): transkript geçmişi soruları düzeldi (kaldığım dersler, 11. dönem, 21 ders), 240 doğru; gereksiz devir ve devirde yanlış katsayı tablosu kaldı.
- **EXP-010:** agent_h1 (tek model Haiku 5.5): cevaplar 56/56, hesap anlamca 24/26 (haftalık saat elle toplanıp yanlış), gereksiz devirler kalktı, ~0,27 $/koşu; gecikme 4,3 → 10,5 sn.
- **EXP-011:** agent_h2: cevaplar 56/56, haftalık saat kodla ama şube çift sayıldı (29, doğrusu 27); ders programı sorgularında gereksiz kod; caching'le koşu 0,27 → 0,12 $.
