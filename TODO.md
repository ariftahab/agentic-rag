# TODO

Yalnız açık işler, sırasıyla. Yapılanların sonuçları config `notlar` alanlarında, `experiments/experiment_log.md`'de;
tarihçe git'te. Agentic yolun işleri GitHub issue'larında (spec #1; açık ticket'lar #14-#17).

## Kaldığımız yer (2026-10-22 akşam)

Durum: agent varsayılanı `agent_v4` (EXP-003); deneyler EXP-002 (agent baseline) ve EXP-003 koşuldu, tool metriği
ikiye ayrıldı (gerekli tool, yan etkisiz). Spec #1 hikâye hikâye kontrol edildi; boşluklar:

- **Sonraki iş, sırayla:** #14 `get_graduation_requirements` tool'u (prompt değişikliği; önce alt küme teşhisi) →
  #15 model karşılaştırması deneyi (onayla) → #16 agent cevaplarında kaynak, 👍/👎 ve /admin → #17 tıklanabilir atıflar.
  #13 kapandı (2026-10-22): web sohbeti transkriptsiz de agent'a gidiyor.
- **#14 teşhis 1 (2026-10-22, agent_v5 = v4 + "genel mezuniyet şartlarında get_graduation_requirements", commit'siz):**
  "Bilgisayar Mühendisliğinden mezun olabilmek için hangi şartlar gerekiyor?" düzeldi (tek tool çağrısı; 244 AKTS,
  zorunlu/seçmeli, yuva kuralı, GNO 2,00, geçer notlar, ADSD ve 2023 istisnası, hepsi kaynaklı; staj ayrıca yazılmadı).
  Kontrol sorularında yan etki: transkript yüklüyken "Genel not ortalamam mezun olmaya yetiyor mu?" analyze_transcript
  yerine yeni tool'u çağırıp transkript yüklemesini istedi (kötüleşme); "Alan dışı seçmeli almadım, engel olur mu?"
  iki kayıt yılı durumunu doğru anlattı ama transkripte bakmadı (2023 girişli için "engel değil" demedi).
- **#14 teşhis 2 (aynı gün, kural daraltıldı: "soru kişiye özel değilse ..., kendi durumunu soruyorsa ve transkript
  yüklüyse önce analyze_transcript"):** mezuniyet şartları sorusu yine tam ve kaynaklı; GNO sorusu düzeldi
  (analyze_transcript, "1,97 < 2,00", EXP-003 ile aynı); alan dışı seçmeli sorusu teşhis 1 ile aynı (kural doğru,
  transkripte bakılmıyor; EXP-003'ten beri açık sorun). Not: setteki beklenen tool mezuniyet şartları sorusunda
  search_regulations olduğu için gerekli tool metriği bu iyileşmeyi hata sayar; tam koşuda anlamca okuma esas.
  EXP-004 koşuldu (answer_acc 0,78 → 0,85; transkript yüklüyken "Alan dışı seçmeli ders almam gerekiyor mu?"
  kötüleşti); #14 kapandı, varsayılan agent_v4'te kaldı (kullanıcı kararı 2026-10-22).
- **ADSD yönlendirme teşhisi 1 (2026-10-22, commit'siz, working tree'de `agent.py`):** get_graduation_requirements
  sonucuna transkript yüklüyse "kendi durumunu soruyorsa analyze_transcript çağır" satırı eklendi. Mezuniyet
  şartları ve GNO soruları değişmedi (doğru). "Alan dışı seçmeli ders almam gerekiyor mu?" (2024 girişli) artık
  analyze_transcript çağırıyor ve dersin alınmadığını söylüyor, ama kayıt yılını bilmediği için "kayıt yılınızı
  kontrol edin" diyor: analyze_transcript raporunda öğrencinin kayıt yılı yazmıyor (yalnız kuralın sonucu).
  "Alan dışı seçmeli almadım, engel olur mu?" (2023 girişli) yine analyze_transcript çağırmadı. Kısmi iyileşme;
  ikinci deneme kullanıcı kararını bekliyor.
- **ADSD teşhisi 2 (aynı gün, commit'siz):** agent_v6 = v5 prompt'u + tool çıktıları: analyze_transcript planın
  altına "Kayıt yılı: 2024-2025 (Alan Dışı Seçmeli Ders bu öğrenci için zorunlu)" yazıyor, yönlendirme satırı
  yalnız v6'da (EXP-004 tekrar üretilebilir). "Alan dışı seçmeli ders almam gerekiyor mu?" (2024 girişli) düzeldi:
  "kayıt yılınıza göre zorunlu, henüz alınmamış". Mezuniyet şartları (stajlar artık ayrı madde) ve GNO soruları
  doğru. "Alan dışı seçmeli almadım, engel olur mu?" (2023 girişli) düzelmedi: analyze_transcript yerine
  search_regulations çağırıyor, iki durumu anlatıp kayıt yılına bırakıyor. 2 deneme doldu; karar kullanıcıda.
  Kullanıcı kararı: bu soru şimdilik bırakıldı; agent_v6 varsayılan yapılmadan commit'lendi.
- **Anlık tool (#18 kapandı, 2026-10-22):** agent_v7 = v6 + request_new_tool (Claude Haiku 5.5, Anthropic sandbox).
  Hızlandırma (2026-10-22, 7ed5aff): dosya şeması görevle veriliyor; Haiku'nun payı haftalık saat sorusunda 24-30 sn,
  AA AKTS sorusunda 10 sn (tek kod turu). Toplam sürenin geri kalanı Gemini'de ve değişken (bir koşuda 98 sn).
  Log (24dc836): `uretilen_kod` görev, Haiku'nun adımları (kod → çıktı → metin), Haiku cevabı, süre, token;
  Gemini süresi = latency_ms − sure_ms. Sonraki adımlar: agent_v7'yi deneyle ölçmek (sorular_v1 bu yolu
  kapsamıyor, mevcut tool'larla cevaplanamayan sorular sorular_v2'ye); run.py'nin make_agent'ı CodeRunner vermiyor.
- **Spec #1 metnini güncelle:** seçmeliler yuva bazlı (her yuva kendi havuzundan, kendi AKTS'si kadar; kullanıcı kararı
  2026-10-22), tool sayısı 6 (+ `confirm_equivalence`, `get_offered_electives`), transkript kaynağı UBYS.
- **Kalan agent hataları (EXP-003):** "Alan dışı seçmeli almadım, engel olur mu?" yalnız mevzuat arıyor (2023 girişliye
  isteğe bağlı olduğunu söylemiyor); "Seçmeli VII fazlam VIII'i kapatmıyor mu?" kural doğru ama transkript sayıları yok;
  açılmayan dersle sorulan what-if'te simulate_courses atlanıp yalnız "açılmıyor" deniyor. Sonraki prompt sürümünde
  alt küme teşhisiyle ele alınır.
- **Yaz okulu sabit cevabı agent yolunda yok:** config'te `fixed_answers: summer_v1` var ama agent kullanmıyor; yaz
  okulu soruları normal mevzuat aramasıyla cevaplanıyor. Sabit cevabın agent'ta gerekip gerekmediğine karar verilecek.
- **Ders programında öğretim elemanı yok:** spec hikâyesi 28 istiyor; PDF'te yalnız 4. sınıfta sütun var
  (`schedule.py`'de kullanılmıyor).
- **Agent log'u geçici olarak tam:** `logs/agent.jsonl` soru, cevap ve tool argümanlarını tutuyor (kullanıcı kararı
  2026-10-21, geliştirme için). Yayına almadan önce gizlilik yeniden ele alınacak (#10'un "log'da transkript içeriği
  yok" ölçütü şu an bilerek sağlanmıyor).
- **Soru setindeki beklentiler:** transkriptsiz mezuniyet sorusunda beklenen tool `analyze_transcript`; agent_v4 tool
  çağırmadan yükleme istiyor (doğru davranış), gerekli tool metriğinde hata sayılıyor. Set değiştirilmez; bir sonraki
  set sürümünde (sorular_v2) düzeltilir. Kota: ücretsiz katmanda dakikalık limit, deneyler `--bekle 15` ile koşulur.
- **Mutasyon testi (cosmic-ray, `graduation.py`):** son koşu yapılmadı (test komutu `-X utf8` ile düzeltildi). Komut:
  `venv/Scripts/cosmic-ray init cosmic-ray.toml <db>` → `exec` → `cr-report`. Koşu sürerken `graduation.py` yerinde
  değişir; başka test/iş koşulmaz. Bugünkü yuva bazlı seçmeli ve what-if kodunu da kapsasın.

## Sıra

Başlangıç 2026-10-21: production EXP-001 = campus-chatbot retrieval hattı, corpus v1 (Mühendislik, 24 belge), tek prompt v1.

1. [x] **Ticket'lar** — #7 what-if, #9 açılan seçmeliler, #11 metrikler kapandı (2026-10-22).
2. [ ] **Model erişilebilirliği (#15)** — `gemini-3.1-flash-lite` 2026-10-21'de sık 503 verdi (3.8-flash da). Seçenekler:
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
4. [x] **Soru seti v1** — 45 soru, 4 sahte transkript (#12, 2026-10-22); EXP-002 ve EXP-003 bununla koşuldu.
5. [ ] **2021 ders içerikleri corpus v2** — soru seti hazır olunca deney olarak.
6. [ ] **Lisansüstü belgeleri** — şimdilik corpus'ta (kullanıcı kararı 2026-10-21).
