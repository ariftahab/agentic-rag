# TODO

Yalnız açık işler, sırasıyla. Yapılanların sonuçları config `notlar` alanlarında, `experiments/experiment_log.md`'de;
tarihçe git'te. Agentic yolun işleri GitHub issue'larında (spec #1, ticket'lar #2-#12).

## Kaldığımız yer (2026-10-21 akşam)

- **Mutasyon testi (cosmic-ray, `graduation.py`):** ilk koşu 658 mutanttan 137'si hayatta; testler eklendi
  (`c5862b4`), ikinci koşuda 55 hayatta (hepsi incelendi: 4 gerçek boşluk testlendi, kalanlar eşdeğer ya da
  `__main__`). **Son koşu yapılmadı:** test komutu `-X utf8` ile düzeltildi (Türkçe pytest çıktısı yüzünden
  öldürülen mutantlar INCOMPETENT görünüyordu). Komut: `venv/Scripts/cosmic-ray init cosmic-ray.toml <db>` →
  `exec` → `cr-report`. Koşu sürerken `graduation.py` yerinde değişir; başka test/iş koşulmaz.
- **#7 what-if:** tasarım onaylı, başlanmadı (subagent ile yapılacak). Bitti tanımı: "'Yapay Zeka ve Görüntü İşleme
  alırsam seçmeli AKTS'm tamamlanır mı?' `simulate_courses` çağrısıyla, elle hesaplanmış AKTS sonucuyla
  cevaplanıyor." Aynı işte küçük düzeltme: `graduation.summary_text` zorunlu dersleri "31 / 40" diye yazıyor, birim
  yok; agent bunu AKTS sandı ("ders" birimi eklenmeli).
- **#10 tarayıcı kontrolü kullanıcıda:** yükleme metni, "Transkript yüklü (45 ders)" etiketi, tool listesi,
  "Kaldır", yenilemede transkriptin gitmesi, yarıyıl düzenli PDF'te hata mesajı.
- Son mutasyon koşusu #7'den sonra (yeni kodu da kapsasın) yapılabilir.

- **Agent log'u geçici olarak tam:** `logs/agent.jsonl` soru, cevap ve tool argümanlarını tutuyor (kullanıcı kararı
  2026-10-21, geliştirme için). Yayına almadan önce gizlilik yeniden ele alınacak (#10'un "log'da transkript içeriği
  yok" ölçütü şu an bilerek sağlanmıyor).

- **Agent modeli:** web ve terminalde agent `gemini-3.5-flash-lite` (`agent.AGENT_LLM_MODEL`, 3.1-flash-lite ile
  50-318 sn sürüyordu, sonra 2-3 sn); klasik RAG production modeli değişmedi. Bir mevzuat sorusu yine 55 sn sürdü:
  adım bazında süre kaydı gerekebilir.
- `agent_v1` prompt'u "OBS'den" diyor; kullanıcı metni UBYS. Sonraki prompt sürümünde (`agent_v2`) düzeltilecek.

## Sıra

Başlangıç 2026-10-21: production EXP-001 = campus-chatbot retrieval hattı, corpus v1 (Mühendislik, 24 belge), tek prompt v1.

1. [ ] **Sıradaki ticket'lar** — #7 what-if, #11 metrikler.
2. [ ] **Model erişilebilirliği** — `gemini-3.1-flash-lite` 2026-10-21'de sık 503 verdi (3.8-flash da). Seçenekler:
       503'te yedek modele geçiş (fallback) ve/veya production modelini değiştirmek (deney). Soru seti hazır olunca:
       `gemini-3.5-flash-lite` ve Claude Haiku 5.5 karşılaştırması (tahmini ~0,001 $/soru).
3. [ ] **Paket sürümleri** — langchain-google-genai 3.0.1 thought signature hatası `agent.py`'de yamalı; 4.x
       `langchain-core` 1.0.4 sabitlemesiyle çakışıyor. En güncel sürümlerle ayrı ortamda test (2026-10-21): `pip check`
       temiz, 35 test geçiyor, retrieval aynı sonucu veriyor, CUDA çalışıyor (torch 2.11.0+cu128, langchain-core 1.6.12,
       langchain-google-genai 4.4.2, transformers 5.20.2, sentence-transformers 6.2.0, huggingface_hub 1.33.0). Agent'ın canlı
       tool turu 4.x'te yamasız çalışıyor (yama kaldırılabilir). Süre: aramalar aynı (~0,4 sn); açılış toplamı
       aynı (~18-20 sn), yalnız reranker yüklemesi 2,7 → 5 sn (transformers 5'in yeni ağırlık yükleme yolu). Önceki
       16,1 / 47,8 sn ölçümleri kurulumdan hemen sonraki soğuk açılışlardı. Yükseltme
       retrieval davranışını değiştirebileceği için deney olarak (yeni index fingerprint'i, sonuçların karşılaştırması).
       **Karar (2026-10-22):** #7 ve #11 bitince, ilk soru seti deneyinden (#12) önce yükseltilir; bütün tez
       deneyleri aynı ortamda koşsun diye. Yükseltmeyle `agent.py`'deki thought signature yaması kaldırılır.
4. [ ] **Soru seti v1** — danışmanlık + transkript senaryoları (#12); baseline EXP-001 ve agent bununla koşulur.
5. [ ] **2021 ders içerikleri corpus v2** — soru seti hazır olunca deney olarak.
6. [ ] **Lisansüstü belgeleri** — şimdilik corpus'ta (kullanıcı kararı 2026-10-21).
