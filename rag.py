"""Getirme (vektör / BM25 / hibrit + isteğe bağlı reranker) ve Gemini ile cevap üretimi.

Tüm davranış bir config.RagConfig ile belirlenir; bkz. experiments/configs/.
"""
import re
import time
import unicodedata
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from rank_bm25 import BM25Okapi

import config
import ingest


def tr_lower(text):
    """Türkçe'ye uygun küçük harf: 'I' -> 'ı', 'İ' -> 'i' (str.lower bunu yanlış yapar)."""
    text = unicodedata.normalize("NFC", text)
    return text.replace("I", "ı").replace("İ", "i").lower()


def tokenize(text):
    # Türkçe eklemeli bir dil; kelimelerin ilk 5 harfini almak basit ama etkili bir kök bulma
    # yöntemi ("geçiş", "geçişi", "geçişler" -> "geçiş").
    return [w[:5] for w in re.findall(r"\w+", tr_lower(text))]


def source_label(doc):
    m = doc.metadata
    if "madde" in m:
        return f"{m['title']}, Madde {m['madde']}, s. {m['page']}"
    if m.get("section"):
        return f"{m['title']}, {m['section']}, s. {m['page']}"
    return f"{m['title']}, s. {m['page']}"


class Retriever:
    """cfg.retrieval'a göre vektör, BM25 ya da ikisinin RRF birleşimi; varsa reranker ile yeniden sıralar."""

    def __init__(self, cfg, store, chunks):
        self.cfg = cfg
        self.store = store
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])
        self.reranker = None
        if cfg.reranker_model:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(cfg.reranker_model, max_length=cfg.reranker_max_length, device=cfg.device)
            if cfg.reranker_dtype == "fp16":
                # Ağırlıklar 16 bit: GPU'da ~3× hızlı, ~1 GB daha az bellek; puanlar fp32'den çok az farklıdır.
                self.reranker.model.half()
    def _vector_ids(self, query, k):
        return [d.metadata["chunk_id"] for d in self.store.similarity_search(query, k=k)]

    def _bm25_ids(self, query, k):
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in ranked[:k] if scores[i] > 0]

    def rank(self, query, depth=10, trace=None):
        """Parçaların sıralı listesini döndürür (en az `depth` uzunluğunda, mümkünse).

        LLM'e bunun ilk final_k elemanı gider; değerlendirme ise sıralamanın tamamına bakar.
        `trace` bir sözlük verilirse ara aşamaların chunk_id listeleri içine yazılır
        ("vector", "bm25", "rrf", "rerank_input", "final"); sonuç bundan etkilenmez.
        """
        cfg = self.cfg
        trace = {} if trace is None else trace
        if cfg.retrieval == "vector":
            ids = trace["vector"] = self._vector_ids(query, max(cfg.vector_k, depth))
        elif cfg.retrieval == "bm25":
            ids = trace["bm25"] = self._bm25_ids(query, max(cfg.bm25_k, depth))
        elif cfg.retrieval == "hybrid":
            # Reciprocal Rank Fusion: iki listede de üst sıralarda olan parçalar öne çıkar.
            trace["vector"] = self._vector_ids(query, cfg.vector_k)
            trace["bm25"] = self._bm25_ids(query, cfg.bm25_k)
            fused = {}
            for ranked in (trace["vector"], trace["bm25"]):
                for rank, chunk_id in enumerate(ranked):
                    fused[chunk_id] = fused.get(chunk_id, 0) + 1 / (cfg.rrf_k + rank)
            ids = trace["rrf"] = sorted(fused, key=fused.get, reverse=True)
        else:
            raise ValueError(f"Bilinmeyen retrieval: {cfg.retrieval}")

        if self.reranker:
            # İlk rerank_candidates aday yeniden sıralanır, geri kalanlar sırasını korur.
            head, tail = ids[:cfg.rerank_candidates], ids[cfg.rerank_candidates:]
            trace["rerank_input"] = list(head)
            # batch_size=8: varsayılan 32 çift tek seferde işlenince 6 GB GPU'da bellek taşıyor (bge-m3 de
            # GPU'da); 30 aday 1,2 sn yerine ~6 sn sürüyor. Batch boyutu puanları değil, yalnız belleği/hızı etkiler.
            scores = self.reranker.predict([(query, self.chunks[i].page_content) for i in head], batch_size=8)
            head = [head[j] for j in sorted(range(len(head)), key=lambda j: scores[j], reverse=True)]
            ids = head + tail
        trace["context_n"] = cfg.final_k  # LLM'e giden parça sayısı
        trace["final"] = list(ids)
        return [self.chunks[i] for i in ids]

    def search(self, query):
        trace = {}
        ranking = self.rank(query, trace=trace)
        return ranking[:trace["context_n"]]


# --- Prompt'lar ---
# Sürümler silinmez; eski deneylerin tekrar üretilebilmesi için yenisi yeni anahtarla eklenir.

# v1: campus-chatbot'un son production prompt'undan (v8_kapsamsiz) türetildi. Corpus yalnız Mühendislik Fakültesi
# öğrencilerine uygulanan belgeler olduğu için fakülte sorma ve kapsam ayırma kuralları çıkarıldı; soran öğrenci
# Mühendislik Fakültesi öğrencisi kabul edilir.
_V1_SYSTEM = """Sen Bolu Abant İzzet Baysal Üniversitesi (BAİBÜ) Mühendislik Fakültesi öğrencilerine yardım eden bir asistansın.

Kurallar:
- Yalnızca aşağıdaki "Bağlam" bölümündeki bilgilere dayanarak cevap ver.
- Cevap bağlamda yoksa yalnızca "{no_answer}" yaz.
- Kullandığın her bilginin sonunda kaynağını köşeli parantezle belirt (bağlamdaki parça etiketlerinde yazan belge adı ve madde).
- Soran kişiyi Mühendislik Fakültesinde lisans öğrencisi olarak kabul et; soru lisansüstü eğitimi soruyorsa lisansüstü
  belgelerdeki kuralı ver.
- Bağlamda hem genel bir kural hem Mühendislik Fakültesine özgü bir kural varsa Mühendislik Fakültesine özgü kuralı esas al.
- Farklı belgeler aynı kuralı veriyorsa bu kuralı tek cümlede, kaynaklarını aynı köşeli parantezde ';' ile ayırarak yaz.
- Belgede bir istisna notu varsa ve soruyla ilgiliyse belirt.
- Bugünün tarihi: {today}. Tarih sorularında gerekirse bu tarihin öncesinde mi sonrasında mı olduğunu belirt.
- Kısa, net ve Türkçe cevap ver.

Bağlam:
{context}"""


def _format_context(docs):
    return "\n\n".join(f"[{source_label(d)}]\n{d.page_content}" for d in docs)


PROMPTS = {
    "v1": {
        "template": ChatPromptTemplate.from_messages([
            ("system", _V1_SYSTEM),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]),
        "format": _format_context,
        "history": True,
    },
}

# citation_style="fikra": atıf kuralı, madde metnindeki fıkra numarasını da istemek üzere değiştirilir (prompt sürümünden
# bağımsız; bütün prompt'larda aynı satır). Panelde fıkrayı vurgulayabilmek için (2026-10-08: 64 atıfın
# hiçbirinde fıkra yoktu). Örnek soyut tutulur: gerçek bir madde/fıkra soru setindeki bir cevabı ele verebilir.
_CITE_RULE = ("- Kullandığın her bilginin sonunda kaynağını köşeli parantezle belirt (bağlamdaki parça etiketlerinde yazan "
              "belge adı ve madde).")
_CITE_RULE_FIKRA = ("- Kullandığın her bilginin sonunda kaynağını köşeli parantezle belirt: bağlamdaki parça etiketinde yazan "
                    "belge adı, madde ve fıkra. Fıkra, madde metninde paragrafın başındaki parantez içindeki sayıdır (\"(2) ...\" ile "
                    "başlayan paragraf ikinci fıkradır). Atıfı [Belge adı, Madde N/F] biçiminde yaz (N madde, F fıkra "
                    "numarası). Bilgi birden fazla fıkradaysa hepsini yaz; madde metninde fıkra numarası yoksa yalnız maddeyi "
                    "yaz.")


def with_citation_style(template, style):
    """Sistem mesajındaki atıf kuralını citation_style'a göre değiştirilmiş yeni bir template döndürür."""
    if style == "madde":
        return template
    system = template.messages[0].prompt.template
    if _CITE_RULE not in system:
        raise ValueError("citation_style='fikra' için prompt'ta atıf kuralı satırı bulunamadı.")
    return ChatPromptTemplate.from_messages([("system", system.replace(_CITE_RULE, _CITE_RULE_FIKRA)),
                                             *template.messages[1:]])


RECITATION_HINT = ("\n\n(Cevabı belge metnini kelimesi kelimesine kopyalamadan, kendi cümlelerinle yaz; kaynak "
                   "atıflarını koru.)")
RECITATION_FALLBACK = ("Bu sorunun cevabı aşağıdaki kaynaklarda yer alıyor, ancak metin otomatik olarak aktarılamadı. "
                       "Lütfen kaynaklara bakın.")

# Yaz öğretimi yönetmeliği yürürlükten kalktı (corpus v4'ten çıkarıldı) ama staj yönergeleri, kimlik kartı yönergesi gibi
# belgelerde eski atıflar duruyor ve model bunlardan eskimiş cevaplar kuruyordu (campus-chatbot EXP-023..027). "yaz dönemi" kapsanmaz:
# yazın yapılan staj soruları belgelerden cevaplanmaya devam eder.
SUMMER_PATTERN = re.compile(r"yaz\s*okul|yaz\s*öğretim")
SUMMER_ANSWER = (f"{config.NO_ANSWER} Yaz öğretimi (yaz okulu) yönetmeliği yürürlükten kaldırıldığı için yaz okulu "
                 "hakkında güncel bir bilgi veremiyorum. Güncel durum için Öğrenci İşleri Daire Başkanlığına başvurabilirsiniz.")

REWRITE_SYSTEM = (
    "Sana bir sohbet geçmişi ve kullanıcının son sorusu verilecek. Son soruyu, geçmişi görmeden de "
    "anlaşılabilecek bağımsız bir soru olarak yeniden yaz (ör. 'peki bunun son günü ne?' -> "
    "'Bahar dönemi ders kayıtlarının son günü ne?'). Soruyu cevaplama; sadece yeniden yazılmış soruyu "
    "döndür. Soru zaten bağımsızsa aynen döndür."
)


def _to_messages(history):
    """{"role": "user"|"assistant", "content": ...} listesini LangChain mesajlarına çevirir (son HISTORY_MESSAGES)."""
    kinds = {"user": HumanMessage, "assistant": AIMessage}
    return [kinds[m["role"]](m["content"]) for m in (history or [])][-config.HISTORY_MESSAGES:]


def token_usage(metadata):
    """Cevap üretiminin token sayıları (RECITATION tekrarı varsa iki çağrının toplamı).

    reasoning: modelin cevaptan önce ürettiği "thinking" token'ları; API raporlamazsa 0. Latency'nin neden değiştiğini
    ayırmak için (daha uzun cevap mı, düşünme mi, yavaş üretim mi).
    """
    out = {"input": 0, "output": 0, "reasoning": 0}
    for m in metadata:
        if not m:
            continue
        out["input"] += m.get("input_tokens", 0)
        out["output"] += m.get("output_tokens", 0)
        out["reasoning"] += (m.get("output_token_details") or {}).get("reasoning", 0)
    return out


class RagBot:
    """Stateless: sohbet geçmişini kendisi tutmaz, her çağrıda dışarıdan alır.

    Böylece aynı bot (ve GPU'daki modeller) birden fazla oturum arasında paylaşılabilir; geçmişi
    çağıran taraf tutar (chat.py'de bir liste, API'de istemci). Geçmiş biçimi:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    """

    def __init__(self, cfg, retriever, llm):
        self.cfg = cfg
        self.retriever = retriever
        self.llm = llm
        self.prompt = PROMPTS[cfg.prompt_version]
        self.template = with_citation_style(self.prompt["template"], cfg.citation_style)
        self.rewrite_chain = ChatPromptTemplate.from_messages([
            ("system", REWRITE_SYSTEM),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]) | llm

    @classmethod
    def from_config(cls, cfg):
        store, chunks, _ = ingest.load_index(cfg)
        # max_retries düşük: kota dolduğunda (429) kullanıcı dakikalarca beklemesin.
        llm = ChatGoogleGenerativeAI(model=cfg.llm_model, temperature=cfg.llm_temperature, max_retries=2)
        return cls(cfg, Retriever(cfg, store, chunks), llm)

    def rewrite(self, question, history=None):
        """Geçmiş varsa (ve prompt geçmiş kullanıyorsa) soruyu bağımsız hale getirir; yoksa aynen döndürür."""
        messages = _to_messages(history)
        if messages and self.prompt["history"]:
            return self.rewrite_chain.invoke({"history": messages, "question": question}).text.strip()
        return question

    def fixed_answer(self, query):
        """Belgelerden cevaplanmaması gereken konular için sabit cevap; yoksa None."""
        text = tr_lower(query)
        if self.cfg.fixed_answers == "summer_v1" and SUMMER_PATTERN.search(text):
            return SUMMER_ANSWER
        return None

    def retrieve(self, question, history=None):
        """Geçmiş varsa soruyu bağımsız hale getirip ilgili parçaları döndürür."""
        query = self.rewrite(question, history)
        return query, self.retriever.search(query)

    def ask(self, question, history=None):
        """Cevap üretir; geçmişi güncellemez (yeni soru-cevabı geçmişe eklemek çağıranın işi)."""
        query = self.rewrite(question, history)
        fixed = self.fixed_answer(query)
        if fixed:
            return {"answer": fixed, "query": query, "docs": [], "generation_ms": 0.0}
        docs = self.retriever.search(query)
        variables = {"input": question, "context": self.prompt["format"](docs)}
        if self.prompt["history"]:
            variables.update({
                "history": _to_messages(history),
                "no_answer": config.NO_ANSWER,
                "today": date.today().strftime("%d.%m.%Y"),
            })
        chain = self.template | self.llm
        started = time.perf_counter()  # yalnız cevap üretimi: retrieval ve rewrite hariç
        message = chain.invoke(variables)
        usage = [message.usage_metadata]
        answer = message.text.strip()
        if not answer and message.response_metadata.get("finish_reason") == "RECITATION":
            # Gemini, cevabın bir metni (ör. yönetmelik maddesini) kelimesi kelimesine tekrar edeceğini düşününce
            # çıktıyı engelleyip boş döndürüyor (campus-chatbot EXP-023: kısmi zamanlı ücret soruları). Bir kez kendi cümleleriyle
            # yazması istenir; yine boşsa kullanıcı boş cevap yerine kaynaklara yönlendirilir.
            message = chain.invoke({**variables, "input": question + RECITATION_HINT})
            usage.append(message.usage_metadata)
            answer = message.text.strip() or RECITATION_FALLBACK
        generation_ms = (time.perf_counter() - started) * 1000
        return {"answer": answer, "query": query, "docs": docs, "generation_ms": generation_ms,
                "usage": token_usage(usage)}
