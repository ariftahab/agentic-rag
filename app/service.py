"""Service katmanı: RAG çekirdeği ile API arasındaki iş mantığı.

Soru sorma, her soru-cevabı kaydetme, geri bildirimi kaydetme ve corpus'u listeleme burada yapılır.
Bu modül FastAPI'yi ya da Gradio'yu bilmez; hangi arayüzden gelirse gelsin her istek aynı şekilde
işlenir ve kaydedilir.

Kayıtlar append-only JSONL dosyalarıdır (her satır bir JSON nesnesi, satırlar sonradan değiştirilmez):
  logs/sohbet.jsonl         her soru-cevap (hata alanları dahil)
  logs/geri_bildirim.jsonl  👍/👎 geri bildirimi; "id" alanıyla sohbet kaydına bağlanır
  logs/degerlendirme.jsonl  elle değerlendirme (cevap etiketi, kaynak doğru mu, hata aşaması); "id" ile bağlı
Aynı cevaba birden fazla geri bildirim / değerlendirme yazılabilir; en son yazılan geçerlidir.
Tek istisna delete(): yöneticinin istediği bir soru-cevap (ör. uygunsuz soru) üç dosyadan da kalıcı olarak silinir.

Agent (transkriptli danışmanlık) istekleri ayrı bir dosyaya, yalnız metadata olarak yazılır:
  logs/agent.jsonl          agent soru-cevapları: soru, cevap, tool çağrıları (argümanlarıyla), süre, hata türü
  logs/anlik_tool.jsonl     Haiku'nun yazıp çalıştırdığı kodlar (soru, kod, çıktı, Haiku'nun cevabı); "id" ile
                            agent.jsonl'e bağlı. İşe yarayan hesaplar buradan kalıcı tool'a çevrilir.
Transkript yükleme hiçbir şey yazmaz: PDF bellekte okunur, sonuç istemciye döner ve unutulur.
"""
import io
import json
import os
import subprocess
import threading
import time
import uuid
from collections import Counter
from datetime import datetime


import config
import graduation
import transcript as transcript_parser
import codeact
from agent import AdvisorAgent

# Branch exp/haiku-agent: web'deki agent tek model Claude Haiku: döngünün tamamı, hazır tool'lar ve anlık kod
# (code_execution) aynı modelde, Gemini yok. agent_h3 = agent_h2 (EXP-011) + duyuru ve akademik personel tool'ları;
# agent_h4 = h3 + personel araması birimi de arar (EXP-014). Klasik RAG (RagBot) production config'iyle aynı kalır.
# agent_h6 = h5 + transkriptten alt küme ortalaması, bütün haftada program araması, şube planı (EXP-015).
WEB_AGENT_PROMPT_VERSION = "agent_h6"
from rag import RagBot, source_label

LOG_DIR = config.ROOT / "logs"
MAX_TRANSCRIPT_BYTES = 5 * 1024 * 1024  # UBYS transkripti ~40 KB; sınır yalnız kötüye kullanıma karşı

# Elle değerlendirme etiketleri (API şemaları da aynı değerleri kullanır).
ANSWER_LABELS = ("dogru", "kismen", "yanlis", "cevapsiz")  # cevapsiz: bilgi yokken doğru şekilde "bilgim yok" dedi
SOURCE_OK = ("evet", "hayir", "emin_degil")                # LLM'e doğru kaynak gitti mi?
ERROR_STAGES = ("rewrite", "retrieval", "generation", "corpus_yok")  # yalnız yanlis / kismen için
# Agent cevaplarında hata zincirin hangi halkasında: tool_secimi = Gemini yanlış/eksik tool seçti ya da argümanı (ör.
# anlık tool görevi) yanlış yazdı; tool_sonucu = tool yanlış veri/hesap döndürdü (arama, plan verisi, ders programı,
# Haiku'nun kodu); generation = tool sonucu doğruydu, Gemini yanlış aktardı; corpus_yok = bilgi hiçbir kaynakta yok.
AGENT_ERROR_STAGES = ("tool_secimi", "tool_sonucu", "generation", "corpus_yok")


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _now():
    return datetime.now().isoformat(timespec="seconds")


class ChatService:
    def __init__(self, bot, experiment_id, log_dir=LOG_DIR, agent=None):
        # bot ve agent dışarıdan verilir (dependency injection): testlerde gerçek modeller yerine sahteleri verilebilir.
        self.bot = bot
        self.agent = agent
        self.experiment_id = experiment_id
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.chat_log = log_dir / "sohbet.jsonl"
        self.feedback_log = log_dir / "geri_bildirim.jsonl"
        self.evaluation_log = log_dir / "degerlendirme.jsonl"
        self.agent_log = log_dir / "agent.jsonl"
        self.code_log = log_dir / "anlik_tool.jsonl"
        self._chunks = {c.metadata["chunk_id"]: c for c in bot.retriever.chunks}
        # Sunucu açılırken bir kez okunur: hangi kod sürümü cevap veriyor? (commit edilmemiş değişiklik varsa "-dirty")
        commit = _git("rev-parse", "--short", "HEAD")
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
        self.commit = f"{commit}-dirty" if commit and dirty else commit
        # FastAPI senkron endpoint'leri ayrı thread'lerde çalıştırır; iki istek aynı anda dosyaya yazıp
        # satırları birbirine karıştırmasın diye yazma işlemi kilitle sıraya sokulur.
        self._lock = threading.Lock()
        self._known_ids = {r["id"] for r in self._read(self.chat_log)}
        # Agent cevapları da geri bildirim, değerlendirme ve silme alır (#16); hata aşamaları türüne göre.
        self._agent_ids = {r["id"] for r in self._read(self.agent_log)}
        self._known_ids |= self._agent_ids

    @classmethod
    def from_experiment(cls, experiment_id=None):
        """Modelleri yükler (yavaş: index + GPU modelleri). Uygulama açılırken bir kez çağrılır."""
        experiment_id = experiment_id or config.PRODUCTION_EXPERIMENT
        cfg = config.RagConfig.from_dict(config.load_experiment(experiment_id)["params"])
        return cls.from_config(cfg, experiment_id)

    @classmethod
    def from_config(cls, cfg, experiment_id):
        """RagBot'u ve agent'ı kurar. Agent RagBot'un Retriever'ını paylaşır: index, bge-m3 ve reranker GPU'ya
        bir kez yüklenir (6 GB GPU'da ikinci kopya sığmaz, sığsa da aynı modeli iki kez tutmanın anlamı yok)."""
        bot = RagBot.from_config(cfg)
        agent = AdvisorAgent(bot.retriever, None, graduation.load_plan(), prompt_version=WEB_AGENT_PROMPT_VERSION,
                             code_runner=codeact.CodeRunner())
        return cls(bot, experiment_id, agent=agent)

    # --- Kayıt yardımcıları ---

    def _append(self, path, record):
        line = json.dumps(record, ensure_ascii=False)
        with self._lock, path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    @staticmethod
    def _read(path):
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _remove_id(self, path, answer_id):
        """path'ten id'si answer_id olan bütün satırları siler; silinen satır sayısını döner.

        Dosya önce geçici bir dosyaya yazılır, sonra os.replace ile eskisinin yerine konur (atomic write):
        yazma yarıda kesilse bile log dosyası ya eski ya yeni hâliyle kalır, yarım kalmaz.
        """
        if not path.exists():
            return 0
        lines = path.read_text(encoding="utf-8").splitlines()
        kept = [line for line in lines if not (line.strip() and json.loads(line)["id"] == answer_id)]
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
        os.replace(tmp, path)
        return len(lines) - len(kept)

    @staticmethod
    def _document_info(doc):
        """Corpus dışı kaynak belge (öğretim planı, ders programı): chunk'ı yok, arayüz PDF'i açar."""
        return {"chunk_id": None, "etiket": doc["title"], "source": doc["source"], "title": doc["title"],
                "page": None, "madde": None}

    @staticmethod
    def _source_info(doc):
        m = doc.metadata
        # source/title/page/madde: arayüz kaynağa link verebilsin ve cevaptaki atıfları eşleştirebilsin diye.
        return {"chunk_id": m["chunk_id"], "etiket": source_label(doc), "source": m["source"],
                "title": m["title"], "page": m.get("page"), "madde": m.get("madde")}

    # --- İş mantığı ---

    def ask(self, question, history=None, session_id=None):
        """Soruyu cevaplar ve kaydeder. Hata olursa da kaydeder, sonra hatayı yukarı iletir."""
        record = {
            "id": uuid.uuid4().hex,
            "zaman": _now(),
            "oturum": session_id,
            "soru": question,
            "rewritten_query": None,
            "gecmis_uzunlugu": len(history or []),
            "cevap": None,
            "kaynaklar": [],
            "config": self.experiment_id,
            "commit": self.commit,
            "latency_ms": None,
            "hata": None,
        }
        started = time.perf_counter()
        try:
            result = self.bot.ask(question, history)
            record["cevap"] = result["answer"]
            if result["query"] != question:  # rewrite yapılmadıysa null kalır
                record["rewritten_query"] = result["query"]
            record["kaynaklar"] = [self._source_info(d) for d in result["docs"]]
            return record
        except Exception as e:
            record["hata"] = f"{type(e).__name__}: {str(e)[:500]}"
            raise
        finally:
            # finally: cevap da dönse hata da fırlasa kayıt yazılır.
            record["latency_ms"] = round((time.perf_counter() - started) * 1000)
            self._append(self.chat_log, record)
            self._known_ids.add(record["id"])

    @staticmethod
    def parse_transcript(data):
        """PDF baytlarını bellekte okur (BytesIO: geçici dosya yok); kişisel bilgisiz transkript sözlüğünü döner.

        Hiçbir şey kaydedilmez. Desteklenmeyen ya da okunamayan dosyada TranscriptFormatError (mesajı kullanıcıya
        gösterilebilir); pdfplumber'ın kendi hataları (PDF değil, bozuk dosya) da bu hataya çevrilir.
        """
        if len(data) > MAX_TRANSCRIPT_BYTES:
            raise transcript_parser.TranscriptFormatError(
                "Dosya transkript için fazla büyük. " + transcript_parser.ASK_FOR_HISTORICAL)
        try:
            return transcript_parser.parse_transcript(io.BytesIO(data))
        except transcript_parser.TranscriptFormatError:
            raise
        except Exception as e:
            raise transcript_parser.TranscriptFormatError(
                "Dosya PDF olarak okunamadı. " + transcript_parser.ASK_FOR_HISTORICAL) from e

    def ask_agent(self, question, history=None, transcript=None, confirmations=None, session_id=None):
        """Agent'a sorar. `confirmations` yerinde güncellenir (confirm_equivalence); sunucu saklamaz, istemciye döner.

        Log'a soru, cevap ve tool çağrıları (argümanlarıyla) yazılır: geliştirme aşamasında cevap kalitesi ve
        süre bu kayıtlardan incelenir (kullanıcı kararı 2026-10-21). Transkriptin kendisi yazılmaz; kişisel bilgiler
        (ad, numara, TC) parser'da ayıklandığı için cevaplarda yok, ama cevaplar GNO ve ders listesi içerir.
        Yayına almadan önce bu kayıt yeniden düşünülecek (TODO). Hata yalnız türüyle tutulur.
        """
        if self.agent is None:
            raise RuntimeError("Agent kurulmamış")
        confirmations = {} if confirmations is None else confirmations
        record = {
            "id": uuid.uuid4().hex,
            "zaman": _now(),
            "oturum": session_id,
            "transkript": transcript is not None,
            "soru": question,
            "cevap": None,
            "gecmis_uzunlugu": len(history or []),
            "tool_lar": [],
            "tur": None,
            "config": self.experiment_id,
            "commit": self.commit,
            "latency_ms": None,
            "hata": None,
        }
        started = time.perf_counter()
        try:
            result = self.agent.ask(question, history, transcript=transcript, confirmations=confirmations)
            record["tool_lar"] = [c["name"] for c in result["tool_calls"]]
            record["tool_cagrilari"] = result["tool_calls"]
            record["cevap"] = result["answer"]
            record["tur"] = result.get("turns")
            record["kaynaklar"] = ([self._source_info(d) for d in result.get("sources", [])]
                                   + [self._document_info(d) for d in result.get("documents", [])])
            if result.get("generated_code"):  # request_new_tool'un ürettiği kod: hangi anlık tool'lar istendi (#18)
                record["uretilen_kod"] = result["generated_code"]
                self._log_code(record, result["generated_code"])
        except Exception as e:
            record["hata"] = type(e).__name__
            raise
        finally:
            record["latency_ms"] = round((time.perf_counter() - started) * 1000)
            self._append(self.agent_log, record)
            self._known_ids.add(record["id"])
            self._agent_ids.add(record["id"])
        return {"id": record["id"], "answer": result["answer"], "tool_calls": result["tool_calls"],
                "sources": record["kaynaklar"], "confirmations": confirmations, "latency_ms": record["latency_ms"]}

    def _log_code(self, record, generated):
        """Kod çalışan her soru için anlik_tool.jsonl'e bir satır: yalnız kod adımları ve çıktıları (tool çağrıları ve ara
        metinler agent.jsonl'de)."""
        codes = [{"kod": step["kod"], "cikti": step["cikti"]}
                 for item in generated for step in item.get("adimlar", []) if step["tip"] == "kod"]
        if codes:
            self._append(self.code_log, {"id": record["id"], "zaman": record["zaman"], "soru": record["soru"],
                                         "transkript": record["transkript"], "kodlar": codes,
                                         "haiku_cevabi": generated[-1].get("haiku_cevabi")})

    def feedback(self, answer_id, rating, note=None):
        """rating: 1 (👍) ya da -1 (👎). Bilinmeyen bir cevap kimliği için KeyError."""
        if answer_id not in self._known_ids:
            raise KeyError(answer_id)
        if rating not in (1, -1):
            raise ValueError("rating 1 ya da -1 olmalı")
        record = {"id": answer_id, "puan": rating, "not": note, "zaman": _now()}
        self._append(self.feedback_log, record)
        return record

    def documents(self):
        """Corpus'taki belgeler ve her birinin chunk sayısı."""
        chunks = self.bot.retriever.chunks
        counts = Counter(c.metadata["source"] for c in chunks)
        first = {}
        for c in chunks:
            first.setdefault(c.metadata["source"], c.metadata)
        return [{"source": s, "title": first[s]["title"], "kind": first[s].get("kind"), "chunk_count": n}
                for s, n in sorted(counts.items())]

    def document_chunks(self, source):
        """Bir belgenin chunk'ları (sırasıyla). Belge yoksa KeyError."""
        rows = [{"chunk_id": c.metadata["chunk_id"], "etiket": source_label(c), "metin": c.page_content}
                for c in self.bot.retriever.chunks if c.metadata["source"] == source]
        if not rows:
            raise KeyError(source)
        return rows

    def chunk(self, chunk_id):
        """Tek bir chunk'ın metni ve kaynağı (kaynak paneli için). Yoksa KeyError."""
        c = self._chunks[chunk_id]
        return {**self._source_info(c), "metin": c.page_content}

    def pdf_path(self, source):
        """Corpus'taki bir belgenin PDF yolu. Corpus'ta olmayan her ad KeyError.

        Kullanıcının verdiği ad doğrudan dosya yoluna eklenmez; yalnız corpus listesindeki adlar kabul edilir
        (whitelist). Böylece "../.env" gibi bir adla data/ dışındaki dosyalara ulaşılamaz (path traversal).
        """
        allowed = {c.metadata["source"] for c in self._chunks.values()}
        if self.agent is not None:  # agent'ın kaynak gösterdiği corpus dışı belgeler (öğretim planı, ders programı)
            allowed |= set(self.agent.documents)
        if source not in allowed:
            raise KeyError(source)
        return config.DATA_DIR / source

    # --- Elle değerlendirme ---

    def evaluate(self, answer_id, answer_label, source_ok, error_stage=None, note=None):
        if answer_id not in self._known_ids:
            raise KeyError(answer_id)
        if answer_label not in ANSWER_LABELS or source_ok not in SOURCE_OK:
            raise ValueError("geçersiz etiket")
        stages = AGENT_ERROR_STAGES if answer_id in self._agent_ids else ERROR_STAGES
        if error_stage is not None and (error_stage not in stages or answer_label not in ("yanlis", "kismen")):
            raise ValueError("hata aşaması yalnız 'yanlis' ve 'kismen' cevaplar için verilir")
        record = {"id": answer_id, "cevap_etiketi": answer_label, "kaynak_dogru": source_ok,
                  "hata_asamasi": error_stage, "not": note, "zaman": _now()}
        self._append(self.evaluation_log, record)
        return record

    def delete(self, answer_id):
        """Bir soru-cevabı ve ona bağlı geri bildirim / değerlendirmeleri kalıcı olarak siler. Geri alınamaz."""
        if answer_id not in self._known_ids:
            raise KeyError(answer_id)
        # Okuma-yazma arasında başka bir istek dosyaya satır eklemesin diye kilit bütün işlem boyunca tutulur.
        with self._lock:
            removed = {path.name: self._remove_id(path, answer_id)
                       for path in (self.chat_log, self.agent_log, self.code_log, self.feedback_log,
                                    self.evaluation_log)}
            self._known_ids.discard(answer_id)
            self._agent_ids.discard(answer_id)
        return removed

    def records(self):
        """Tüm soru-cevaplar (yeniden eskiye), her birinin son geri bildirimi ve son değerlendirmesiyle."""
        # Dosyalar sırayla okunduğu için sözlükte aynı id'nin en son satırı kalır ("en son yazılan geçerli").
        feedback = {r["id"]: r["puan"] for r in self._read(self.feedback_log)}
        evaluation = {r["id"]: r for r in self._read(self.evaluation_log)}
        rows = [{**r, "tur": "rag", "tool_lar": []} for r in self._read(self.chat_log)]
        # 2026-10-21'deki ilk agent kayıtları yalnız metadata tutuyordu (soru/cevap yok): değerlendirilecek bir şey yok.
        rows += [self._agent_row(r) for r in self._read(self.agent_log) if "soru" in r]
        rows.sort(key=lambda r: r["zaman"], reverse=True)
        return [{**self._complete_sources(r), "geri_bildirim": feedback.get(r["id"]),
                 "degerlendirme": evaluation.get(r["id"])} for r in rows]

    @staticmethod
    def _agent_row(r):
        """Agent kaydını klasik kaydın alanlarına çevirir (yönetici sayfası ikisini tek listede gösterir)."""
        return {"id": r["id"], "zaman": r["zaman"], "oturum": r["oturum"], "soru": r["soru"], "rewritten_query": None,
                "cevap": r["cevap"], "kaynaklar": r.get("kaynaklar", []), "config": r["config"], "commit": r["commit"],
                "latency_ms": r["latency_ms"], "hata": r["hata"], "tur": "agent", "tool_lar": r["tool_lar"]}

    def _complete_sources(self, record):
        """Kaynaklara source/title/page/madde eklenmeden önce yazılmış kayıtları gösterim için tamamlar.

        chunk_id yalnız aynı index içinde anlamlıdır; bu yüzden yalnız kayıt şu anki config ile üretildiyse
        tamamlanır. Dosyadaki kayıt değişmez (append-only).
        """
        if record["config"] != self.experiment_id:
            return record
        sources = [s if "source" in s or s["chunk_id"] not in self._chunks
                   else {**s, **self._source_info(self._chunks[s["chunk_id"]])} for s in record["kaynaklar"]]
        return {**record, "kaynaklar": sources}

    def summary(self):
        rows = self.records()
        answered = [r for r in rows if not r["hata"]]
        evaluated = [r["degerlendirme"] for r in answered if r["degerlendirme"]]
        labels = Counter(e["cevap_etiketi"] for e in evaluated)
        return {
            "toplam": len(rows),
            "agent": sum(r["tur"] == "agent" for r in rows),
            "hatali_istek": len(rows) - len(answered),
            "degerlendirilen": len(evaluated),
            "cevap_etiketleri": {k: labels.get(k, 0) for k in ANSWER_LABELS},
            # "dogru" + "cevapsiz" doğru davranıştır; kismen ayrı raporlanır.
            "dogruluk": round((labels["dogru"] + labels["cevapsiz"]) / len(evaluated), 3) if evaluated else None,
            "kaynak_dogru": dict(Counter(e["kaynak_dogru"] for e in evaluated)),
            "hata_asamalari": dict(Counter(e["hata_asamasi"] for e in evaluated if e["hata_asamasi"])),
            "geri_bildirim": {"olumlu": sum(r["geri_bildirim"] == 1 for r in rows),
                              "olumsuz": sum(r["geri_bildirim"] == -1 for r in rows)},
        }
