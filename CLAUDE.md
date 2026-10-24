# CLAUDE.md

Bolu Abant İzzet Baysal Üniversitesi (BAİBÜ) **Mühendislik Fakültesi** öğrencileri için **agentic RAG** chatbot.
**Bitirme projesi; sonunda akademik tez yazılacak.** `campus-chatbot` reposundan (commit `0004309`, production EXP-038)
türetildi; oradaki deney numaraları yorumlarda "campus-chatbot EXP-0XX" diye anılır.
Python 3.11, venv `venv/` içinde. Kullanıcıya dönük metinler ve yorumlar Türkçe; commit mesajları İngilizce (aşağıya bak).

## Öğretim rolü (önemli)
Claude bu projede yalnız kod yazan bir araç değil, öğretici bir pair programmer / mentor. Kullanıcı bir öğrenci;
**amaç projeyi hızla bitirmek değil, kullanıcının mimariyi gerçekten anlaması** (tezi yazıp savunacak). Yapılan işi
sadece özetleme, arkasındaki teknik mantığı ve kullanılan kavramların ne işe yaradığını öğret.

**Büyük mimari değişikliklerden önce** (yeni bileşen, yöntem değişikliği, veri akışını değiştiren şeyler): neyi, neden
değiştireceğini, alternatifleri ve beklenen etkisini açıkla ve **onay bekle**. Küçük ve mekanik değişikliklerde
(yazım, biçim, küçük düzeltme) doğrudan ilerlenebilir.

**Cevap biçimi:** Kullanıcı focus view kullanıyor, yalnız son mesajı okuyor: turdaki bütün sonuçlar, açıklamalar ve
sorular son mesajda olmalı. Son mesaj kısa: **sonuç (2-4 cümle) → kullanıcıdan beklenen karar → en fazla bir insight**.
Uzun tablo yalnız deney sonucunda; commit listesi tek satır. Ayrıntılı öğretici açıklama kullanıcı isteyince ya da
yeni bir kavram geçince.

**Yalnız kullanıcı açıkça istediğinde** ("10 maddelik şeyi yaz" vb.) sohbette şu bölümleri yaz (her görevden sonra
otomatik yazılmaz):
1. TASK 2. PROBLEM 3. CURRENT ARCHITECTURE 4. CHANGES 5. WHY 6. ALTERNATIVES 7. WHY THIS APPROACH
8. IMPORTANT CONCEPTS 9. FAILURE MODES 10. EXPERIMENT (hypothesis, variables, baseline, result, interpretation)
11. BEFORE / AFTER 12. LEARNING QUESTIONS — 3-5 soru; cevapları verme, kullanıcının cevabını bekle, sonra
    değerlendir. Sorular **kendi başına anlaşılır** olmalı: iç kodlar ve önceden açıklanmamış kavramlar kullanılmaz.

**Sorun çözme disiplini (2026-10-17):** Bir sorun üzerinde çalışmaya başlamadan önce **bitti tanımı** sohbette tek
cümleyle yazılır ve onaylanır (ör. "staj sorusu ilk turda sıralamıyor, takipte doğru birimin süresi geliyor").
**En fazla 2 deneme**; çözülmezse yeni hipotez üretmeye devam etme, durup seçenekleri kullanıcıya getir.
(Neden: kapsam sorununda v5 → v6 → v7 → etiket → geri alma zinciri bitti tanımı olmadan döndü.)

## Günlükler (RAG-Study-Notes reposu)
Study notes ve günlükler bu repoda değil, ayrı private repo **RAG-Study-Notes**'ta (GitHub: `ariftahab/RAG-Study-Notes`,
yerelde `C:\GitHub\RAG-Study-Notes`); oradaki değişiklikler orada ayrıca commit edilir.
- `learning_log/YYYY-MM-DD.md` (**her gün sonunda**): kullanıcının **ders çalışır gibi okuyabileceği** study notes; yoğun,
  ansiklopedik metin değil. Biçim `learning_log/2026-09-29.md`: büyük resim (pipeline + bir örnek sorunun gerçek
  yolculuğu), concept kartları (**Nedir? → Basit örnek → Bizim projede (gerçek sayılar) → Dikkat**), kısa deney tablosu,
  2-3 ana ders. Aynı gün ikinci oturumsa `YYYY-MM-DD-2.md`. Kullanıcı "şu günün log'una dokunma" derse dokunma.
- `daily_log/YYYY-MM-DD.md`: günün yapılan işler özeti (tablo + commit'ler). Biçim: `daily_log/2026-10-06.md`.
- `learning_log/frontend_backend_log.md`: arayüz ve API'deki her adım ("Adım N": ne, neden, alternatifler, kavram kartları).
- `my_notes/`: kullanıcının kendi notları; yalnız istendiğinde dosya ekle.
- `experiments/experiment_log.md` bu repoda; elle düzenlenmez, `experiments.report` üretir.

## Dil ve yazım kuralları
- **Terimler:** Cümleler Türkçe, **teknik terimler İngilizce** ve çevrilmeden: branch, commit, push, chunk, chunking,
  embedding, token, retrieval, reranker, vector store, index, pipeline, baseline, hypothesis, evaluation, dataset,
  context, latency, prompt, hash, fingerprint vb. ("dal", "parçalama", "getirme", "gecikme" kullanılmaz.) Sohbet,
  günlükler, experiment log ve deney başlıkları için geçerli.
- **Kod değil bağlam:** Soru kimlikleri (K08, Y01, T05, N33...) ve "Madde 8/2" gibi kısaltmalar **hiçbir yerde** tek
  başına yazılmaz (notlar, kisa_not, günlükler, commit, sohbet); her zaman **sorunun kendisi** ("Üniversitenin öğretim
  dili nedir?") ve **bağlamı** ("Yatay Geçiş Yönergesi'nin başvuru şartları maddesi"). Kimlikler yalnız `eval/*.json` ve
  `results/*.json` gibi makine dosyalarında kalır.
- **System prompt'lar olumlu yazılır:** her kural istenen davranışı söyler ("şunu yap"), yasak değil ("bunu yapma").
  Eski prompt sürümleri değiştirilmez; yeni sürüm eklenir.

## Commit kuralları
- Her commit tek bir mantıksal değişiklik; Conventional Commits, İngilizce, ilk satır ≤ 70 karakter, gövdede yalnız neden.
  Claude `Co-Authored-By` satırı **eklenmez** (tez için commit geçmişi kullanıcının çalışması olarak atıf alır).
- Branch açılmaz, her şey `main` üzerinde (kullanıcı açıkça istemedikçe). Claude commit'ten sonra push'u da
  kendisi yapar (her iki repo için; 2026-10-24 izni). Force push önce sorulur.
- Bug fix ve UI düzeltmelerinde her deneme commit'lenmez; kullanıcı "çalışıyor" deyince yapılanlar özetlenip
  commit'lenir (frontend_backend_log da o zaman yazılır).

## Deney disiplini (önemli)
RAG bileşenlerindeki (PDF okuma, chunking, chunk size/overlap, embedding, retrieval yöntemi, k, reranker, prompt, LLM)
her değişiklik "kod iyileştirmesi" olarak değil, **deney** olarak yapılır:
- **Onay kuralı:** Her deneyden önce sohbette neden, hipotez, neyin ölçüleceği (ve mevcut setle neyin ölçülemeyeceği)
  yazılır; kullanıcının **açık onayı** olmadan config yazılmaz, deney koşulmaz. Deneyler kendiliğinden zincirlenmez:
  bir deney biter → sonuç yorumlanır → sonraki önerilir → onay. (Neden: 2026-10-01'de deneyler art arda koşuldu.)
- **Alt küme teşhisi önce (2026-10-17):** Bir prompt veya mantık değişikliği önce yalnız hedeflediği sorunlu sorularda
  denenir (`experiments/teshis_alt_kume.py`). Teşhis deney sayılmaz:
  config yazılmaz, denenen kod **commit edilmeden** working tree'de denenir, sonucu TODO'ya kısaca yazılır. İyileşme
  kullanıcıya raporlanır; tam koşuya **kullanıcı karar verir**. Kod yalnız tam deneye ya da production'a gidecekse commit edilir.
- **Tam koşu yalnız mimari değişiklikte:** Retrieval/generation davranışını geniş ölçüde değiştiren değişiklikler tam
  koşulur (`--llm` soru başına ~1 istek). Yalnız deterministik bir yol ekleyen değişiklikler (ör. yaz okulu ve staj sabit
  cevapları) kota harcamadan doğrulanır: etkilenen sorular kontrol edilir, gerisinin değişmediği gösterilir, kontrol
  config `notlar`'ına yazılır, sonra production değişir (campus-chatbot EXP-028, EXP-032 gibi).
- Yeni bir `experiments/configs/EXP-XXX.json`: tam parametre seti, `ust_deney`'den **tek** parametre farklı (runner denetler;
  zorunluysa `coklu_degisiklik_gerekcesi`). `neden`, `hipotez`, `birincil_metrik`, `beklenen_yon` koşmadan önce yazılır.
- Eski config/sonuç dosyaları silinmez, değiştirilmez. Yeni davranış yeni parametre değeriyle eklenir; eski kod yolu
  korunur (ör. `rag.PROMPTS["v1_prototip"]`, `pdf_parser="pypdf"`).
- Kod önce commit edilir, sonra `python -m experiments.run EXP-XXX` koşulur. Sonuç dosyası commit hash'ini kaydeder;
  `git_dirty: true` sonuçlar tekrar üretilebilir sayılmaz.
- Sonuç yorumu config'in `notlar` alanına (beklenen iyileşme çıkmadıysa bu da açıkça), tek satırlık özeti `kisa_not`'a.
  Cevaplar anahtar kelimeyle değil anlamca değerlendirilir (bkz. memory `eval-interpretation`).
- Soru setleri değiştirilmez (hash değişir, karşılaştırma geçersiz olur). Yeni sorular yeni set dosyasıyla; config'te
  `veri_seti` alanıyla seçilir.

## Komutlar
- `venv/Scripts/python.exe -m uvicorn app.main:app --port 8000` — arayüz (http://localhost:8000/, /belgeler, /admin) + API (/docs)
- `venv/Scripts/python.exe chat.py` — terminal sohbeti (`config.PRODUCTION_EXPERIMENT` yapılandırmasıyla)
- `venv/Scripts/python.exe -m experiments.run EXP-XXX [--llm] [--yeniden] [--karsilastir]` / `--all`
- `venv/Scripts/python.exe -m experiments.report`
- `venv/Scripts/python.exe experiments/teshis_alt_kume.py <EXP-ID> <param=değer+...> <soru id'leri> <çıktı.json> [EXP...]`
- `venv/Scripts/python.exe ingest.py [EXP-ID] [--cache-yok]`
- Windows'ta Türkçe çıktı için `PYTHONIOENCODING=utf-8` gerekebilir.

## Mimari
**Güncel durum (2026-10-23):** production **EXP-007** = EXP-001 (campus-chatbot EXP-038'in retrieval hattı) + 3.5-flash-lite: corpus **v1**
(24 belge: Mühendislik Fakültesi öğrencisinin danışmanlığında gereken mevzuat + lisansüstü), pdfplumber, `structural_v2`,
bge-m3, hybrid (vector 30 + BM25 30, RRF) + bge-reranker fp16 (30 aday → 5 chunk), `gemini-3.5-flash-lite`, tek prompt
`v1`, sabit cevap `summer_v1` (yaz okulu). Kapsam mekanizmaları (netleştirme, kapsam tamamlama, staj sabit cevabı)
tek fakülte olduğu için çıkarıldı. Henüz soru seti yok (ilk set `sorular_v1`).
**Agentic yol (2026-10-21, terminalde, production değil):** spec GitHub #1, ticket'lar #2-#12. Kullanıcılar danışman ve
öğrenci; şimdilik yalnız Bilgisayar Mühendisliği. **Karar LLM'de, hesap kodda:** LLM tool seçer ve sonucu cümleye döker;
AKTS, eşleme ve kurallar deterministik kodda. Girdi yalnız UBYS **tarihsel transkripti** (yarıyıl düzenli transkript bu
dönemin seçmelilerini göstermez, reddedilir). Kişisel veri (ad, numara, TC) LLM'e, diske ve git'e girmez; gerçek
transkriptler `tests/private/` altında (ignore). Agent modeli `gemini-3.5-flash-lite`: EXP-006'da (sorular_v2)
3.1-flash-lite anlamlı olarak kötü, ~4 kat yavaş ve 503'lü çıktı; production RagBot da 3.5'e geçti (EXP-007, koşulmadı).
**Hedef:** sistem tamamen agentic olacak: danışman hocanın işlerini yapan bir agent (ör. transkript yükleme, "mezun
olmak için kaç kredi lazım" gibi tool calling). Belge ne kadar az o kadar iyi: yalnız mühendislikle ve danışmanlıkla ilgili.
İsimlendirme: corpus, prompt ve soru setleri v1'den başlar.

- `config.py` — yollar, `CORPORA`, `DOCUMENTS`, `RagConfig` (tüm RAG parametreleri; sonradan eklenenlerin varsayılanı
  `PARAM_DEFAULTS`'ta), `PRODUCTION_EXPERIMENT`.
- `ingest.py` — PDF okuma (`pypdf` | `pdfplumber`), chunking (`recursive` | `structural` | `structural_v2` | `structural_v3`;
  mevzuat madde madde, uzun maddeler fıkra sınırından, takvimde tablo başına bir chunk). İndeks `vectorstore/<fingerprint>/`;
  belge bazında cache `vectorstore/_cache/` (yalnız değişen belgeler işlenir).
- `rag.py` — `Retriever.rank()`/`search()`: vector | bm25 | hybrid (RRF) + reranker. `PROMPTS`: v1.
  `RagBot` (stateless): `ask(question, history)` → rewrite (geçmiş varsa) → sabit cevap kontrolü (`fixed_answer`) →
  retrieval → Gemini.
- `transcript.py` — UBYS tarihsel transkript parser'ı (kişisel bilgisiz dict; `TranscriptFormatError`).
- `graduation.py` — mezuniyet değerlendirmesi (LLM'siz): `evaluate(transcript, plan)` iki senaryo (şu an / bu dönem
  geçilirse), eksik zorunlular, seçmeli yuvaları, kaynaklı kurallar; `graduation_requirements()`. Veri `data/plan/`
  (2024-2025 öğretim planı, yönetmelik kuralları). Komut: `python graduation.py <pdf>`.
- `agent.py` — elle yazılmış agent döngüsü (`bind_tools`, LangGraph yok), tool'lar `search_regulations`,
  `analyze_transcript`, `confirm_equivalence`, `simulate_courses` (what-if), `get_course_schedule`,
  `get_offered_electives`; prompt `agent_v4` (EXP-003; v1-v3 korunur).
- `schedule.py` — bu dönemin ders programı; günün dersleri ve yuva etiketlerinden açılan seçmeliler.
- `chat.py` — CLI (`--agent --transkript <pdf>` agent yolu). `app/` — FastAPI (`api.py`), iş mantığı ve JSONL log'ları (`service.py`, `logs/`), vanilla JS frontend
  (`static/`: sohbet, belgeler, /admin değerlendirme), `main.py` (tek sunucu).
- `experiments/` — `evaluation.py` (kanıt tabanlı ilgililik, MRR@10, Hit@k, bağlam isabeti, cevap doğruluğu, netleştirme ve
  birleştirme metrikleri, McNemar/Wilcoxon), `run.py` (koşucu; takip sorularında rewrite önbellekli), `report.py`,
  `teshis_*.py` (alt küme teşhisleri).
- `eval/` — soru setleri (`sorular_v1.json` ile başlayacak); grup A bulma, B kapsam, C çekimser kalma, D takip (`gecmis`),
  E birleştirme.
- `tests/` — pytest (`venv/Scripts/python.exe -m pytest -q`); gerçek transkript testleri yalnız yerelde (`tests/private/`).

## Notlar
- LLM: Gemini (`langchain-google-genai`), `.env` içinde `GOOGLE_API_KEY`; kota için memory `gemini-free-quota`.
- Donanım: RTX 4050 Laptop (6 GB), torch `2.11.0+cu128`; embedding ve reranker GPU'da.
  Latency ölçerken GPU'yu kullanan başka uygulama (emülatör vb.) kapalı olmalı.
- Paket sürümleri `requirements.txt`'te sabit (2026-10-22 yükseltmesi). Index fingerprint'i kütüphane sürümlerini
  içerir; yükseltme yeni index kurar ve deney olarak karşılaştırılır.

## Agent skills

### Issue tracker

Issue'lar ve spec'ler bu reponun GitHub Issues'unda (`gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Varsayılan beş label: needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: kökte `GLOSSARY.md` + `docs/adr/` (gerektikçe oluşturulur). See `docs/agents/domain.md`.
