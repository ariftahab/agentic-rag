# TODO

Yalnız açık işler, sırasıyla. Yapılanların sonuçları config `notlar` alanlarında, `experiments/experiment_log.md`'de;
tarihçe git'te. Agentic yolun işleri GitHub issue'larında (spec #1, ticket'lar #2-#12).

## Sıra

Başlangıç 2026-10-21: production EXP-001 = campus-chatbot retrieval hattı, corpus v1 (Mühendislik, 24 belge), tek prompt v1.

1. [ ] **Ticket'ları kapat** — #2, #3, #4, #5 kodda tamam (#5 `gemini-3.5-flash-lite` ile doğrulandı: transkript sorusu
       3,5 sn, mevzuat sorusu 2,9 sn); kullanıcı onayıyla kapatılacak.
2. [ ] **Sıradaki ticket'lar** — #6 Erasmus eşlemesi (İngilizce ad + kelime örtüşmesi; adlar `data/BILG_2021.pdf`'te,
       bulunamayanlar çeviri), #7 what-if, #8 ders programı (`data/2026_27_Güz_Lisans_v2.pdf`), #10 arayüz, #11 metrikler.
3. [ ] **analyze_transcript cevabında eksik bilgi** — agent "bu dönem geçilirse 187 AKTS" ve "seçmeliden 40 AKTS açık"
       bilgisini cevaba koymadı; tool sonucu LLM için kısa bir özete indirilsin (~11 KB rapor hem yavaş hem pahalı).
       Önce alt küme teşhisi.
4. [ ] **Model erişilebilirliği** — `gemini-3.1-flash-lite` 2026-10-21'de sık 503 verdi (3.8-flash da). Seçenekler:
       503'te yedek modele geçiş (fallback) ve/veya production modelini değiştirmek (deney). Soru seti hazır olunca:
       `gemini-3.5-flash-lite` ve Claude Haiku 5.5 karşılaştırması (tahmini ~0,001 $/soru).
5. [ ] **Paket sürümleri** — langchain-google-genai 3.0.1 thought signature hatası `agent.py`'de yamalı; 4.x
       `langchain-core` 1.0.4 sabitlemesiyle çakışıyor. En güncel sürümlerle ayrı ortamda test (2026-10-21): `pip check`
       temiz, 35 test geçiyor, retrieval aynı sonucu veriyor, CUDA çalışıyor (torch 2.11.0+cu128, langchain-core 1.6.12,
       langchain-google-genai 4.4.2, transformers 5.20.2, sentence-transformers 6.2.0, huggingface_hub 1.33.0). Açık:
       index yükleme + arama 16,1 sn (mevcut 8,6 sn), agent'ın canlı tool turu 4.x'te ve yamasız denenmedi. Yükseltme
       retrieval davranışını değiştirebileceği için deney olarak (yeni index fingerprint'i, sonuçların karşılaştırması).
6. [ ] **Soru seti v1** — danışmanlık + transkript senaryoları (#12); baseline EXP-001 ve agent bununla koşulur.
7. [ ] **2021 ders içerikleri corpus v2** — soru seti hazır olunca deney olarak.
8. [ ] **Lisansüstü belgeleri** — şimdilik corpus'ta (kullanıcı kararı 2026-10-21).
