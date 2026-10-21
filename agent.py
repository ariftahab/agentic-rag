"""Agent döngüsü: LLM hangi tool'a ihtiyaç olduğuna karar verir, kod tool'u çalıştırır, sonuç LLM'e geri verilir.

Döngü elle yazılır (LangGraph yok): langchain'in `bind_tools`'u Gemini'ye tool tanımlarını verir.
    LLM → (tool istekleri) → tool'lar çalışır → ToolMessage'lar geri verilir → LLM → ... → nihai cevap
Tur sınırı (MAX_TURNS) LLM'in sonsuza kadar tool istemesini engeller.

Karar LLM'de, hesap kodda: AKTS toplama ve ders eşleme graduation modülünde yapılır; LLM yalnız tool seçer
ve sonucu cümleye döker. RagBot gibi stateless: geçmiş ve parse edilmiş transkript her çağrıda dışarıdan
verilir. Transkript LLM'in argümanı değildir; oturumda durur ve analyze_transcript onu kendisi okur.
LLM'e transkriptin kendisi değil, yalnız değerlendirme raporu gider (kişisel bilgi zaten parser'da ayıklanır).
"""
import base64
from datetime import date

import langchain_google_genai.chat_models as _genai_chat
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI

import config
import graduation
import ingest
import schedule
from rag import Retriever, _format_context, _to_messages, token_usage

MAX_TURNS = 5  # bir soru için en fazla LLM çağrısı (tool turları + nihai cevap)
REQUEST_TIMEOUT_S = 180  # tek bir LLM isteği için en fazla bekleme


# --- langchain-google-genai 3.0.x düzeltmesi: thought signature ---
# Gemini 3 modelleri her tool isteğine (function call) bir "thought signature" ekler; tool sonucu geri verilirken
# bu imzanın aynen geri gönderilmesi zorunludur, yoksa API 400 döner. langchain-google-genai 3.0.1 (ve 3.0.3) cevapta
# yalnız tool isteği varsa imzayı atıyor; ayrıca geri gönderirken imzayı content listesindeki sıraya göre
# tool isteğiyle eşliyor. Bu sarmalayıcı, tool isteği içeren cevaplarda content'i "i. tool isteğinin imzası i.
# sırada" olacak şekilde yeniden kurar. Tool isteği olmayan cevaplara (RagBot'un bütün cevapları) dokunmaz.
# Paket 4.x'e yükseltilince (langchain-core sabitlemesi gözden geçirilerek) kaldırılabilir.
def _keep_function_call_signatures(parse):
    def wrapper(candidate, *args, **kwargs):
        message = parse(candidate, *args, **kwargs)
        if not message.tool_calls or kwargs.get("streaming"):
            return message
        blocks = []
        for part in candidate.content.parts:
            if not part.function_call:
                continue
            signature = getattr(part, "thought_signature", None)
            blocks.append({"type": "function_call_signature", "signature": base64.b64encode(signature).decode("ascii")}
                          if signature else {"type": "text", "text": ""})
        message.content = blocks
        return message
    return wrapper


if not getattr(_genai_chat._parse_response_candidate, "_keeps_signatures", False):
    _genai_chat._parse_response_candidate = _keep_function_call_signatures(_genai_chat._parse_response_candidate)
    _genai_chat._parse_response_candidate._keeps_signatures = True

# --- System prompt'lar ---
# Sürümler silinmez; değişiklik yeni anahtarla eklenir (rag.PROMPTS gibi). Kurallar olumlu yazılır.
_AGENT_V1_SYSTEM = """Sen Bolu Abant İzzet Baysal Üniversitesi (BAİBÜ) Mühendislik Fakültesi için bir danışmanlık asistanısın. Sorulara danışman hocalar da öğrenciler de sorabilir.

Kurallar:
- Mevzuat sorularında (staj, ders kaydı, sınavlar, yatay geçiş, mezuniyet şartları vb.) search_regulations tool'unu kullan ve cevabı yalnız dönen parçalara dayandır.
- Kullandığın her bilginin sonunda kaynağını köşeli parantezle belirt (parça etiketlerinde yazan belge adı ve madde).
- Transkriptle ilgili sorularda (eksik dersler, tamamlanan AKTS, mezuniyet durumu) analyze_transcript tool'unu kullan; sayıları raporda yazdığı gibi aktar ve öğretim planını kaynak olarak göster.
- Bütün sayıları tool sonuçlarından al ve olduğu gibi aktar; hesabı tool'lar yapar.
- Transkript gerektiğinde, OBS'den indirilen tarihsel transkripti yüklemesini iste; bu dönem alınan dersler (seçmeliler dahil) yalnız tarihsel transkriptte görünür.
- Ders seçimi ve tercih sorularında kuralı ve hesabı ver, kararı için soranı danışman hocasına yönlendir.
- Hitapta nötr ol: aynı cevap danışman hocaya da öğrenciye de uysun; transkriptten "transkriptteki öğrenci" gibi üçüncü kişi ifadeleriyle bahset.
- Tool'lardan dönen bilgi soruyu cevaplamaya yetmiyorsa yalnızca "{no_answer}" yaz ve ilgili birime başvurmayı öner.
- Bugünün tarihi: {today}.
- Kısa, net ve Türkçe cevap ver."""

AGENT_PROMPTS = {"agent_v1": _AGENT_V1_SYSTEM}
AGENT_PROMPT_VERSION = "agent_v1"

NO_TRANSCRIPT = ("Bu oturumda yüklenmiş bir transkript yok. Kullanıcıdan OBS'den indirdiği tarihsel transkripti "
                 "yüklemesini iste.")
TURN_LIMIT_ANSWER = "Bu soruyu belirlenen adım sayısı içinde cevaplayamadım. Soruyu daha dar sorabilir misiniz?"


def _same_name(a, b):
    return graduation.name_similarity(a, b) == 1.0


def make_tools(retriever, plan, transcript, confirmations, today):
    """Bu çağrıya (oturuma) bağlı tool'lar. `transcript` ve `confirmations` closure'da durur; LLM'in görebileceği
    argümanlar değildir. confirm_equivalence `confirmations` sözlüğünü yerinde günceller (çağıran taraf saklar)."""

    def search_regulations(query: str) -> str:
        """Üniversitenin yönetmelik ve yönergelerinde (staj, ders kaydı, sınavlar, mezuniyet, yatay geçiş, lisansüstü vb.)
        arama yapar ve en ilgili parçaları kaynak etiketleriyle döndürür.

        Args:
            query: Aranacak konu, Türkçe ve bağımsız bir soru ya da ifade olarak (ör. "staj başvurusu nasıl yapılır").
        """
        return _format_context(retriever.search(query))

    def analyze_transcript() -> str:
        """Oturumda yüklü transkripti öğretim planıyla karşılaştırır: şu an geçilen AKTS, bu dönemin dersleri
        geçilirse ulaşılacak AKTS ve eksik zorunlu dersler (durumlarıyla). Transkript yüklenmemişse bunu bildirir."""
        if transcript is None:
            return NO_TRANSCRIPT
        return graduation.summary_text(graduation.evaluate(transcript, plan, confirmations=confirmations), plan)

    def confirm_equivalence(transcript_course: str, plan_course: str) -> str:
        """Kullanıcının teyidini kaydeder: transkriptteki plan dışı bir dersin (Erasmus, intibak) öğretim planındaki
        hangi derse sayıldığı. Kayıttan sonra güncel transkript analizini döndürür.

        Args:
            transcript_course: Transkriptteki plan dışı dersin adı, analizde yazdığı gibi (ör. "Microcomputer Systems").
            plan_course: Öğretim planındaki dersin Türkçe ya da İngilizce adı (ör. "Mikroişlemciler"); kullanıcı plandaki
                hiçbir derse karşılık gelmediğini söylediyse boş metin.
        """
        if transcript is None:
            return NO_TRANSCRIPT
        course = next((c for c in transcript["dersler"] if _same_name(c["ad"], transcript_course)), None)
        if course is None:
            return f'Transkriptte "{transcript_course}" adlı ders bulunamadı. Adı analizde yazdığı gibi kullan.'
        target = None
        if plan_course.strip():
            target = next((c for c in graduation.plan_courses(plan)
                           if _same_name(c["ad"], plan_course) or _same_name(c.get("ad_en", ""), plan_course)), None)
            if target is None:
                return (f'Öğretim planında "{plan_course}" adlı ders bulunamadı. Kullanıcıdan plandaki dersin adını '
                        "iste ya da eksik dersler listesindeki adlardan birini kullan.")
        confirmations[course["kod"]] = target["kod"] if target else None
        saved = f"{course['ad']} → {target['ad']}" if target else f"{course['ad']}: plandaki bir derse karşılık gelmiyor"
        return f"Kaydedildi ({saved}). Güncel analiz:\n\n" + analyze_transcript()

    def get_course_schedule(gun: str = "bugün", sinif: int = 0, dersler: list[str] | None = None) -> str:
        """Bilgisayar Mühendisliği bu dönemki haftalık ders programından bir günün derslerini saat, şube ve derslikle
        döndürür; resmi tatili, hafta sonunu ve ders dönemi dışını belirtir. Sınıf ve ders verilmezse transkriptteki
        bu dönemin dersleri ("benim derslerim") kullanılır.

        Args:
            gun: "bugün", "yarın", bir gün adı (ör. "perşembe") ya da YYYY-AA-GG tarihi.
            sinif: Sınıf (1-4); bilinmiyorsa 0.
            dersler: Ders adları (ör. ["Yapay Zeka"]); bilinmiyorsa boş.
        """
        try:
            day = schedule.resolve_day(gun, today)
        except ValueError as e:
            return str(e)
        codes = None
        if not sinif and not dersler:
            if transcript is None:
                return ("Transkript yüklü değil ve sınıf ya da ders verilmedi. Kullanıcıya kaçıncı sınıf olduğunu ya da "
                        "hangi dersleri aldığını sor.")
            codes = schedule.current_course_codes(transcript)
        data = schedule.load_schedule()
        text = schedule.schedule_text(schedule.lookup(data, day, grade=sinif or None, courses=dersler, codes=codes),
                                      data)
        if codes is not None:
            text += ("\nDersler transkriptteki bu dönemin derslerinden alındı; öğrencinin şubesi bilinmediği için "
                     "bütün şubeler listelendi.")
        return text

    tools = (search_regulations, analyze_transcript, confirm_equivalence, get_course_schedule)
    return [StructuredTool.from_function(f) for f in tools]


class AdvisorAgent:
    """Stateless: geçmişi ve transkripti çağıran taraf tutar (chat.py'de değişkenler, ileride API'de istemci)."""

    def __init__(self, retriever, llm, plan, prompt_version=AGENT_PROMPT_VERSION):
        self.retriever = retriever
        self.llm = llm
        self.plan = plan
        self.system = AGENT_PROMPTS[prompt_version]

    @classmethod
    def from_config(cls, cfg):
        """Production RagBot'un retrieval hattı ve modeliyle (aynı config) kurulur."""
        store, chunks, _ = ingest.load_index(cfg)
        # timeout: tool sonucu içeren isteklerde API bazen dakikalarca cevap vermiyor (2026-10-21); süresiz beklemek
        # yerine hata verilir, chat.py bunu yakalayıp sohbete devam eder.
        llm = ChatGoogleGenerativeAI(model=cfg.llm_model, temperature=cfg.llm_temperature, max_retries=2,
                                     timeout=REQUEST_TIMEOUT_S)
        return cls(Retriever(cfg, store, chunks), llm, graduation.load_plan())

    def ask(self, question, history=None, transcript=None, on_tool_call=None, confirmations=None):
        """Cevap üretir. `on_tool_call(name, args)` verilirse her tool çağrısında çağrılır (terminalde göstermek için).
        `confirmations` oturumun eşdeğerlik teyitleridir ({plan dışı ders kodu: plan ders kodu | None}); transkript
        gibi çağıran taraf saklar, confirm_equivalence yerinde günceller.

        Dönüş: {"answer", "tool_calls": [{"name", "args"}], "turns", "usage"}.
        """
        today = date.today()
        tools = make_tools(self.retriever, self.plan, transcript, {} if confirmations is None else confirmations, today)
        by_name = {t.name: t for t in tools}
        llm = self.llm.bind_tools(tools)
        system = self.system.format(no_answer=config.NO_ANSWER, today=today.strftime("%d.%m.%Y"))
        messages = [SystemMessage(system), *_to_messages(history), HumanMessage(question)]

        calls, usage = [], []
        for turn in range(1, MAX_TURNS + 1):
            message = llm.invoke(messages)
            usage.append(message.usage_metadata)
            messages.append(message)
            if not message.tool_calls:
                return {"answer": message.text.strip(), "tool_calls": calls, "turns": turn,
                        "usage": token_usage(usage)}
            for call in message.tool_calls:
                calls.append({"name": call["name"], "args": call["args"]})
                if on_tool_call:
                    on_tool_call(call["name"], call["args"])
                tool = by_name.get(call["name"])
                try:
                    result = tool.invoke(call["args"]) if tool else f"Bilinmeyen tool: {call['name']}"
                except Exception as e:  # hatalı argüman vb.: LLM düzeltebilsin diye hata metni geri verilir
                    result = f"Tool hatası: {type(e).__name__}: {e}"
                messages.append(ToolMessage(content=result, tool_call_id=call["id"], name=call["name"]))
        return {"answer": TURN_LIMIT_ANSWER, "tool_calls": calls, "turns": MAX_TURNS, "usage": token_usage(usage)}
