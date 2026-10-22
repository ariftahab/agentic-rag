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

- **agent_v4 teşhisi (2026-10-22, deney değil, kod working tree'de):** prompt'a oturumun transkript durumu, "kendi
  durumunu sorana önce analyze_transcript" ve "confirm_equivalence yalnız kullanıcı eşleme bildirince" kuralları. EXP-002'de
  transkript farkındalığı hatası olan 6 soruda anlamca doğru cevap 2 → 5: stajlarım tamam mı, ortalamam yetiyor mu,
  Erasmus dersi sayılmış mı (yan etkili teyit kalktı), alan dışı seçmeli almam gerekiyor mu düzeldi. Kalan: "alan dışı
  seçmeli almadım, engel olur mu?" yine yalnız mevzuat aradı (2023 girişliye "engel olabilir" dedi); "Seçmeli VII fazlam
  VIII'i kapatmıyor mu?" kural doğru ama sayısız. Kontroller bozulmadı; transkriptsiz soruda artık tool çağırmadan yükleme
  istiyor (doğru davranış, ama setin beklenen tool'u analyze_transcript olduğu için tool metriğinde hata sayılır).

## Sıra

Başlangıç 2026-10-21: production EXP-001 = campus-chatbot retrieval hattı, corpus v1 (Mühendislik, 24 belge), tek prompt v1.

1. [x] **Ticket'lar** — #7 what-if, #9 açılan seçmeliler, #11 metrikler kapandı (2026-10-22).
2. [ ] **Model erişilebilirliği** — `gemini-3.1-flash-lite` 2026-10-21'de sık 503 verdi (3.8-flash da). Seçenekler:
       503'te yedek modele geçiş (fallback) ve/veya production modelini değiştirmek (deney). Soru seti hazır olunca:
       `gemini-3.5-flash-lite` ve Claude Haiku 5.5 karşılaştırması (tahmini ~0,001 $/soru).
3. [x] **Paket sürümleri (2026-10-22)** — torch 2.11.0+cu128, transformers 5.20.2, sentence-transformers 6.2.0,
       langchain-core 1.6.13, langchain-google-genai 4.4.2 (`requirements.txt`); eski ortam `venv-old/` (silinebilir).
       Index fingerprint'i ve belge cache anahtarı artık kütüphane sürümlerini içeriyor. Karşılaştırma (eski / yeni
       ortam, aynı kod): 685 chunk aynı, embedding'ler aynı (en büyük fark 3e-8), 20 sorgunun 20'sinde ilk sonuç, 19'unda
       ilk 5 aynı (birinde 2. ve 3. yer değişti, fp16 reranker). Arama 0,45 → 0,37 sn; açılış 8,7 → 14 sn (reranker
       yüklemesi). Thought signature yaması kaldırıldı; agent'ın çok tool'lu turu canlıda yamasız çalıştı.
       **Not:** langchain-google-genai 4.x, `gemini-3.5-flash-lite`'ın sabit örnekleme kullandığını ve `temperature`'ı
       yok saydığını bildiriyor; agent deneylerinde `llm_temperature` etkisiz.
4. [ ] **Soru seti v1** — danışmanlık + transkript senaryoları (#12); baseline EXP-001 ve agent bununla koşulur.
5. [ ] **2021 ders içerikleri corpus v2** — soru seti hazır olunca deney olarak.
6. [ ] **Lisansüstü belgeleri** — şimdilik corpus'ta (kullanıcı kararı 2026-10-21).
