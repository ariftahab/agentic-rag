# TODO

Yalnız açık işler, sırasıyla. Yapılanların sonuçları config `notlar` alanlarında, `experiments/experiment_log.md`'de;
tarihçe git'te. Agentic yolun işleri GitHub issue'larında (spec #1, ticket'lar #2-#12).

## Sıra

Başlangıç 2026-10-21: production EXP-001 = campus-chatbot retrieval hattı, corpus v1 (Mühendislik, 24 belge), tek prompt v1.

1. [ ] **Sıradaki ticket'lar** — #6 Erasmus eşlemesi (İngilizce ad + kelime örtüşmesi; adlar `data/BILG_2021.pdf`'te,
       bulunamayanlar çeviri), #7 what-if, #8 ders programı (`data/2026_27_Güz_Lisans_v2.pdf`), #10 arayüz, #11 metrikler.
2. [ ] **analyze_transcript cevabında eksik bilgi** — agent "bu dönem geçilirse 187 AKTS" ve "seçmeliden 40 AKTS açık"
       bilgisini cevaba koymadı. Teşhis (2026-10-21, `gemini-3.5-flash-lite`, "Mezun olmak için neyim eksik?" 5'er koşu):
       mevcut JSON rapor (10,7 bin karakter) 187'yi 1/5, açık 40 AKTS'yi 0/5 verdi; kodla üretilen kısa özet (3 bin
       karakter, iki senaryo başta) 187'yi 5/5, açık 40'ı açıkça 2/5 verdi (diğerlerinde 24/64), giriş token'ı 5.939 →
       2.461; iki senaryo kuralı eklenmiş prompt ikisini 5/5 verdi ama GNO'yu 5/5 düşürdü, ikinci senaryonun eksik
       derslerini 4/5 belirsiz yazdı. Karar kullanıcıda (özet + prompt birlikte denensin mi).
3. [ ] **Model erişilebilirliği** — `gemini-3.1-flash-lite` 2026-10-21'de sık 503 verdi (3.8-flash da). Seçenekler:
       503'te yedek modele geçiş (fallback) ve/veya production modelini değiştirmek (deney). Soru seti hazır olunca:
       `gemini-3.5-flash-lite` ve Claude Haiku 5.5 karşılaştırması (tahmini ~0,001 $/soru).
4. [ ] **Paket sürümleri** — langchain-google-genai 3.0.1 thought signature hatası `agent.py`'de yamalı; 4.x
       `langchain-core` 1.0.4 sabitlemesiyle çakışıyor. En güncel sürümlerle ayrı ortamda test (2026-10-21): `pip check`
       temiz, 35 test geçiyor, retrieval aynı sonucu veriyor, CUDA çalışıyor (torch 2.11.0+cu128, langchain-core 1.6.12,
       langchain-google-genai 4.4.2, transformers 5.20.2, sentence-transformers 6.2.0, huggingface_hub 1.33.0). Agent'ın canlı
       tool turu 4.x'te yamasız çalışıyor (yama kaldırılabilir). Süre: aramalar aynı (~0,4 sn); açılış toplamı
       aynı (~18-20 sn), yalnız reranker yüklemesi 2,7 → 5 sn (transformers 5'in yeni ağırlık yükleme yolu). Önceki
       16,1 / 47,8 sn ölçümleri kurulumdan hemen sonraki soğuk açılışlardı. Yükseltme
       retrieval davranışını değiştirebileceği için deney olarak (yeni index fingerprint'i, sonuçların karşılaştırması).
5. [ ] **Soru seti v1** — danışmanlık + transkript senaryoları (#12); baseline EXP-001 ve agent bununla koşulur.
6. [ ] **2021 ders içerikleri corpus v2** — soru seti hazır olunca deney olarak.
7. [ ] **Lisansüstü belgeleri** — şimdilik corpus'ta (kullanıcı kararı 2026-10-21).
