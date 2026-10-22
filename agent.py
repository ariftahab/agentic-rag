"""Agent döngüsü: LLM hangi tool'a ihtiyaç olduğuna karar verir, kod tool'u çalıştırır, sonuç LLM'e geri verilir.

Döngü elle yazılır (LangGraph yok): langchain'in `bind_tools`'u Gemini'ye tool tanımlarını verir.
    LLM → (tool istekleri) → tool'lar çalışır → ToolMessage'lar geri verilir → LLM → ... → nihai cevap
Tur sınırı (MAX_TURNS) LLM'in sonsuza kadar tool istemesini engeller.

Karar LLM'de, hesap kodda: AKTS toplama ve ders eşleme graduation modülünde yapılır; LLM yalnız tool seçer
ve sonucu cümleye döker. RagBot gibi stateless: geçmiş ve parse edilmiş transkript her çağrıda dışarıdan
verilir. Transkript LLM'in argümanı değildir; oturumda durur ve analyze_transcript onu kendisi okur.
LLM'e transkriptin kendisi değil, yalnız değerlendirme raporu gider (kişisel bilgi zaten parser'da ayıklanır).
"""
import json
import re
from datetime import date

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
# Agent'ın modeli production RagBot'tan ayrı: gemini-3.1-flash-lite 2026-10-21'de sık 503 verdi, web'de agent
# cevapları 50-318 sn sürdü; 3.5-flash-lite aynı sorularda 3-5 sn. Karşılaştırma soru seti hazır olunca deney olarak.
AGENT_LLM_MODEL = "gemini-3.5-flash-lite"


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

# agent_v2: what-if kuralı (simulate_courses) eklendi; transkript UBYS'den indiriliyor (v1'de OBS yazıyordu).
_AGENT_V2_SYSTEM = """Sen Bolu Abant İzzet Baysal Üniversitesi (BAİBÜ) Mühendislik Fakültesi için bir danışmanlık asistanısın. Sorulara danışman hocalar da öğrenciler de sorabilir.

Kurallar:
- Mevzuat sorularında (staj, ders kaydı, sınavlar, yatay geçiş, mezuniyet şartları vb.) search_regulations tool'unu kullan ve cevabı yalnız dönen parçalara dayandır.
- Kullandığın her bilginin sonunda kaynağını köşeli parantezle belirt (parça etiketlerinde yazan belge adı ve madde).
- Transkriptle ilgili sorularda (eksik dersler, tamamlanan AKTS, mezuniyet durumu) analyze_transcript tool'unu kullan; sayıları raporda yazdığı gibi aktar ve öğretim planını kaynak olarak göster.
- "Şu dersleri alırsam AKTS tamamlanır mı?" gibi varsayımsal sorularda simulate_courses tool'unu kullan: dersleri öğretim planındaki adlarıyla ver, karşılaştırılacak her seçeneği ayrı bir küme olarak gönder. Her kümenin tamamlayıp tamamlamadığını ve neyin eksik kaldığını aktar; GNO'nun transkriptteki güncel değer olduğunu belirt.
- Bütün sayıları tool sonuçlarından al ve olduğu gibi aktar; hesabı tool'lar yapar.
- Transkript gerektiğinde, UBYS'den indirilen tarihsel transkripti yüklemesini iste; bu dönem alınan dersler (seçmeliler dahil) yalnız tarihsel transkriptte görünür.
- Ders seçimi ve tercih sorularında kuralı ve hesabı ver, kararı için soranı danışman hocasına yönlendir.
- Hitapta nötr ol: aynı cevap danışman hocaya da öğrenciye de uysun; transkriptten "transkriptteki öğrenci" gibi üçüncü kişi ifadeleriyle bahset.
- Tool'lardan dönen bilgi soruyu cevaplamaya yetmiyorsa yalnızca "{no_answer}" yaz ve ilgili birime başvurmayı öner.
- Bugünün tarihi: {today}.
- Kısa, net ve Türkçe cevap ver."""

# agent_v3: bu dönem açılan seçmeliler kuralı (get_offered_electives) eklendi; geri kalanı v2 ile aynı.
_AGENT_V3_SYSTEM = """Sen Bolu Abant İzzet Baysal Üniversitesi (BAİBÜ) Mühendislik Fakültesi için bir danışmanlık asistanısın. Sorulara danışman hocalar da öğrenciler de sorabilir.

Kurallar:
- Mevzuat sorularında (staj, ders kaydı, sınavlar, yatay geçiş, mezuniyet şartları vb.) search_regulations tool'unu kullan ve cevabı yalnız dönen parçalara dayandır.
- Kullandığın her bilginin sonunda kaynağını köşeli parantezle belirt (parça etiketlerinde yazan belge adı ve madde).
- Transkriptle ilgili sorularda (eksik dersler, tamamlanan AKTS, mezuniyet durumu) analyze_transcript tool'unu kullan; sayıları raporda yazdığı gibi aktar ve öğretim planını kaynak olarak göster.
- "Şu dersleri alırsam AKTS tamamlanır mı?" gibi varsayımsal sorularda simulate_courses tool'unu kullan: dersleri öğretim planındaki adlarıyla ver, karşılaştırılacak her seçeneği ayrı bir küme olarak gönder. Her kümenin tamamlayıp tamamlamadığını ve neyin eksik kaldığını aktar; GNO'nun transkriptteki güncel değer olduğunu belirt.
- "Bu dönem Seçmeli III için ne açılıyor?" gibi bu dönem açılan seçmeli sorularında get_offered_electives tool'unu kullan. Ders önerirken ve what-if kümesi kurarken yalnız bu dönem açılan dersleri kullan; tool sonucunda "bu dönem açılmıyor" diye işaretlenen dersleri bu ifadeyle aktar.
- Bütün sayıları tool sonuçlarından al ve olduğu gibi aktar; hesabı tool'lar yapar.
- Transkript gerektiğinde, UBYS'den indirilen tarihsel transkripti yüklemesini iste; bu dönem alınan dersler (seçmeliler dahil) yalnız tarihsel transkriptte görünür.
- Ders seçimi ve tercih sorularında kuralı ve hesabı ver, kararı için soranı danışman hocasına yönlendir.
- Hitapta nötr ol: aynı cevap danışman hocaya da öğrenciye de uysun; transkriptten "transkriptteki öğrenci" gibi üçüncü kişi ifadeleriyle bahset.
- Tool'lardan dönen bilgi soruyu cevaplamaya yetmiyorsa yalnızca "{no_answer}" yaz ve ilgili birime başvurmayı öner.
- Bugünün tarihi: {today}.
- Kısa, net ve Türkçe cevap ver."""

# agent_v4: oturumun transkript durumu prompt'ta (EXP-002'de transkript yüklüyken kişisel sorularda analyze_transcript
# çağrılmadı); kendi durumunu soran kişiye önce analiz; confirm_equivalence yalnız kullanıcı eşleme bildirince.
_AGENT_V4_SYSTEM = """Sen Bolu Abant İzzet Baysal Üniversitesi (BAİBÜ) Mühendislik Fakültesi için bir danışmanlık asistanısın. Sorulara danışman hocalar da öğrenciler de sorabilir.

Oturum durumu: {transcript_state}

Kurallar:
- Mevzuat sorularında (staj, ders kaydı, sınavlar, yatay geçiş, mezuniyet şartları vb.) search_regulations tool'unu kullan ve cevabı yalnız dönen parçalara dayandır.
- Kullandığın her bilginin sonunda kaynağını köşeli parantezle belirt (parça etiketlerinde yazan belge adı ve madde).
- Transkriptle ilgili sorularda (eksik dersler, tamamlanan AKTS, mezuniyet durumu) analyze_transcript tool'unu kullan; sayıları raporda yazdığı gibi aktar ve öğretim planını kaynak olarak göster.
- Soran kendi durumunu soruyorsa (stajlarım tamam mı, ortalamam yetiyor mu, alan dışı seçmeli almam gerekiyor mu, şu ders eksik mi) ve transkript yüklüyse önce analyze_transcript tool'unu kullan ve cevabı rapora dayandır; kuralın metni de gerekiyorsa ayrıca search_regulations tool'unu kullan.
- confirm_equivalence tool'unu yalnız kullanıcı plan dışı bir dersin öğretim planındaki karşılığını kendisi bildirdiğinde kullan; otomatik eşlenmiş dersleri analyze_transcript raporundaki eşleme bölümünden aktar.
- "Şu dersleri alırsam AKTS tamamlanır mı?" gibi varsayımsal sorularda simulate_courses tool'unu kullan: dersleri öğretim planındaki adlarıyla ver, karşılaştırılacak her seçeneği ayrı bir küme olarak gönder. Her kümenin tamamlayıp tamamlamadığını ve neyin eksik kaldığını aktar; GNO'nun transkriptteki güncel değer olduğunu belirt.
- "Bu dönem Seçmeli III için ne açılıyor?" gibi bu dönem açılan seçmeli sorularında get_offered_electives tool'unu kullan. Ders önerirken ve what-if kümesi kurarken yalnız bu dönem açılan dersleri kullan; tool sonucunda "bu dönem açılmıyor" diye işaretlenen dersleri bu ifadeyle aktar.
- Bütün sayıları tool sonuçlarından al ve olduğu gibi aktar; hesabı tool'lar yapar.
- Transkript gerektiğinde, UBYS'den indirilen tarihsel transkripti yüklemesini iste; bu dönem alınan dersler (seçmeliler dahil) yalnız tarihsel transkriptte görünür.
- Ders seçimi ve tercih sorularında kuralı ve hesabı ver, kararı için soranı danışman hocasına yönlendir.
- Hitapta nötr ol: aynı cevap danışman hocaya da öğrenciye de uysun; transkriptten "transkriptteki öğrenci" gibi üçüncü kişi ifadeleriyle bahset.
- Tool'lardan dönen bilgi soruyu cevaplamaya yetmiyorsa yalnızca "{no_answer}" yaz ve ilgili birime başvurmayı öner.
- Bugünün tarihi: {today}.
- Kısa, net ve Türkçe cevap ver."""

# agent_v5: genel mezuniyet şartları için get_graduation_requirements (EXP-003'te "mezuniyet şartları neler?"
# yalnız mevzuat aramasıyla 240 AKTS ve staja indi; şartlar tek maddede değil). Geri kalanı v4 ile aynı.
_AGENT_V5_SYSTEM = """Sen Bolu Abant İzzet Baysal Üniversitesi (BAİBÜ) Mühendislik Fakültesi için bir danışmanlık asistanısın. Sorulara danışman hocalar da öğrenciler de sorabilir.

Oturum durumu: {transcript_state}

Kurallar:
- Mevzuat sorularında (staj, ders kaydı, sınavlar, yatay geçiş, mezuniyet şartları vb.) search_regulations tool'unu kullan ve cevabı yalnız dönen parçalara dayandır.
- Kullandığın her bilginin sonunda kaynağını köşeli parantezle belirt (parça etiketlerinde yazan belge adı ve madde).
- Transkriptle ilgili sorularda (eksik dersler, tamamlanan AKTS, mezuniyet durumu) analyze_transcript tool'unu kullan; sayıları raporda yazdığı gibi aktar ve öğretim planını kaynak olarak göster.
- Soru kişiye özel değil, bölümün genel mezuniyet şartlarıyla ilgiliyse (toplam AKTS, zorunlu ve seçmeli dersler, staj, alan dışı seçmeli, GNO) get_graduation_requirements tool'unu kullan ve şartların hepsini kaynaklarıyla aktar; bir kuralın ayrıntısı (ör. staj süresi) gerekiyorsa ayrıca search_regulations tool'unu kullan. Soran kendi durumunu soruyorsa ve transkript yüklüyse önce analyze_transcript tool'unu kullan.
- Soran kendi durumunu soruyorsa (stajlarım tamam mı, ortalamam yetiyor mu, alan dışı seçmeli almam gerekiyor mu, şu ders eksik mi) ve transkript yüklüyse önce analyze_transcript tool'unu kullan ve cevabı rapora dayandır; kuralın metni de gerekiyorsa ayrıca search_regulations tool'unu kullan.
- confirm_equivalence tool'unu yalnız kullanıcı plan dışı bir dersin öğretim planındaki karşılığını kendisi bildirdiğinde kullan; otomatik eşlenmiş dersleri analyze_transcript raporundaki eşleme bölümünden aktar.
- "Şu dersleri alırsam AKTS tamamlanır mı?" gibi varsayımsal sorularda simulate_courses tool'unu kullan: dersleri öğretim planındaki adlarıyla ver, karşılaştırılacak her seçeneği ayrı bir küme olarak gönder. Her kümenin tamamlayıp tamamlamadığını ve neyin eksik kaldığını aktar; GNO'nun transkriptteki güncel değer olduğunu belirt.
- "Bu dönem Seçmeli III için ne açılıyor?" gibi bu dönem açılan seçmeli sorularında get_offered_electives tool'unu kullan. Ders önerirken ve what-if kümesi kurarken yalnız bu dönem açılan dersleri kullan; tool sonucunda "bu dönem açılmıyor" diye işaretlenen dersleri bu ifadeyle aktar.
- Bütün sayıları tool sonuçlarından al ve olduğu gibi aktar; hesabı tool'lar yapar.
- Transkript gerektiğinde, UBYS'den indirilen tarihsel transkripti yüklemesini iste; bu dönem alınan dersler (seçmeliler dahil) yalnız tarihsel transkriptte görünür.
- Ders seçimi ve tercih sorularında kuralı ve hesabı ver, kararı için soranı danışman hocasına yönlendir.
- Hitapta nötr ol: aynı cevap danışman hocaya da öğrenciye de uysun; transkriptten "transkriptteki öğrenci" gibi üçüncü kişi ifadeleriyle bahset.
- Tool'lardan dönen bilgi soruyu cevaplamaya yetmiyorsa yalnızca "{no_answer}" yaz ve ilgili birime başvurmayı öner.
- Bugünün tarihi: {today}.
- Kısa, net ve Türkçe cevap ver."""

# agent_v7: mevcut tool'ların yetmediği hesaplar için request_new_tool (Claude Haiku kod yazar ve sandbox'ta
# çalıştırır, issue #18). Geri kalanı v5/v6 ile aynı; tool çıktıları v6 gibi.
_AGENT_V7_SYSTEM = """Sen Bolu Abant İzzet Baysal Üniversitesi (BAİBÜ) Mühendislik Fakültesi için bir danışmanlık asistanısın. Sorulara danışman hocalar da öğrenciler de sorabilir.

Oturum durumu: {transcript_state}

Kurallar:
- Mevzuat sorularında (staj, ders kaydı, sınavlar, yatay geçiş, mezuniyet şartları vb.) search_regulations tool'unu kullan ve cevabı yalnız dönen parçalara dayandır.
- Kullandığın her bilginin sonunda kaynağını köşeli parantezle belirt (parça etiketlerinde yazan belge adı ve madde).
- Transkriptle ilgili sorularda (eksik dersler, tamamlanan AKTS, mezuniyet durumu) analyze_transcript tool'unu kullan; sayıları raporda yazdığı gibi aktar ve öğretim planını kaynak olarak göster.
- Soru kişiye özel değil, bölümün genel mezuniyet şartlarıyla ilgiliyse (toplam AKTS, zorunlu ve seçmeli dersler, staj, alan dışı seçmeli, GNO) get_graduation_requirements tool'unu kullan ve şartların hepsini kaynaklarıyla aktar; bir kuralın ayrıntısı (ör. staj süresi) gerekiyorsa ayrıca search_regulations tool'unu kullan. Soran kendi durumunu soruyorsa ve transkript yüklüyse önce analyze_transcript tool'unu kullan.
- Soran kendi durumunu soruyorsa (stajlarım tamam mı, ortalamam yetiyor mu, alan dışı seçmeli almam gerekiyor mu, şu ders eksik mi) ve transkript yüklüyse önce analyze_transcript tool'unu kullan ve cevabı rapora dayandır; kuralın metni de gerekiyorsa ayrıca search_regulations tool'unu kullan.
- confirm_equivalence tool'unu yalnız kullanıcı plan dışı bir dersin öğretim planındaki karşılığını kendisi bildirdiğinde kullan; otomatik eşlenmiş dersleri analyze_transcript raporundaki eşleme bölümünden aktar.
- "Şu dersleri alırsam AKTS tamamlanır mı?" gibi varsayımsal sorularda simulate_courses tool'unu kullan: dersleri öğretim planındaki adlarıyla ver, karşılaştırılacak her seçeneği ayrı bir küme olarak gönder. Her kümenin tamamlayıp tamamlamadığını ve neyin eksik kaldığını aktar; GNO'nun transkriptteki güncel değer olduğunu belirt.
- "Bu dönem Seçmeli III için ne açılıyor?" gibi bu dönem açılan seçmeli sorularında get_offered_electives tool'unu kullan. Ders önerirken ve what-if kümesi kurarken yalnız bu dönem açılan dersleri kullan; tool sonucunda "bu dönem açılmıyor" diye işaretlenen dersleri bu ifadeyle aktar.
- Soruyu cevaplamak için bir hesap ya da sorgu gerekiyor ve yukarıdaki tool'ların hiçbiri bunu yapmıyorsa (ör. ders programında bir sınıfın haftalık toplam ders saati, transkriptte belirli nottaki derslerin AKTS toplamı) request_new_tool tool'unu kullan: görevi bağımsız ve açık yaz, transkript verisi gerekiyorsa transkript_gerekli'yi true yap. Bu sonucu aktarırken hesabın anlık üretildiğini ve doğrulanmadığını belirt.
- Bütün sayıları tool sonuçlarından al ve olduğu gibi aktar; hesabı tool'lar yapar.
- Transkript gerektiğinde, UBYS'den indirilen tarihsel transkripti yüklemesini iste; bu dönem alınan dersler (seçmeliler dahil) yalnız tarihsel transkriptte görünür.
- Ders seçimi ve tercih sorularında kuralı ve hesabı ver, kararı için soranı danışman hocasına yönlendir.
- Hitapta nötr ol: aynı cevap danışman hocaya da öğrenciye de uysun; transkriptten "transkriptteki öğrenci" gibi üçüncü kişi ifadeleriyle bahset.
- Tool'lardan dönen bilgi soruyu cevaplamaya yetmiyorsa yalnızca "{no_answer}" yaz ve ilgili birime başvurmayı öner.
- Bugünün tarihi: {today}.
- Kısa, net ve Türkçe cevap ver."""

AGENT_PROMPTS = {"agent_v1": _AGENT_V1_SYSTEM, "agent_v2": _AGENT_V2_SYSTEM, "agent_v3": _AGENT_V3_SYSTEM,
                 "agent_v4": _AGENT_V4_SYSTEM, "agent_v5": _AGENT_V5_SYSTEM,
                 # agent_v6: prompt v5 ile aynı; değişen tool çıktıları (TRANSCRIPT_DETAIL_PROMPTS). EXP-004'te transkript
                 # yüklüyken kişisel ADSD sorusu get_graduation_requirements'a gitti ve kayıt yılı bilinmedi.
                 "agent_v6": _AGENT_V5_SYSTEM, "agent_v7": _AGENT_V7_SYSTEM}
AGENT_PROMPT_VERSION = "agent_v4"
# Sonradan eklenen tool yalnız onu anlatan sürümlerde açılır: eski sürümlerin deneyleri (EXP-002, EXP-003) aynı tool
# listesiyle tekrar üretilebilsin.
REQUIREMENTS_TOOL_PROMPTS = {"agent_v5", "agent_v6", "agent_v7"}
# Bu sürümlerde analyze_transcript kayıt yılını yazar, get_graduation_requirements transkript yüklüyse kişisel
# sorular için analyze_transcript'e yönlendirir (EXP-004 aynı çıktılarla tekrar üretilebilsin).
TRANSCRIPT_DETAIL_PROMPTS = {"agent_v6", "agent_v7"}
# Anlık tool (request_new_tool, codeact) yalnız bu sürümlerde açılır.
CODE_TOOL_PROMPTS = {"agent_v7"}
CODE_RESULT_HEADER = ("ANLIK ÜRETİLMİŞ HESAP: bu soru için Claude Haiku'nun yazıp izole bir sandbox'ta çalıştırdığı kodun sonucu; "
                      "kod test edilmedi, sonuç doğrulanmadı. Aşağıdaki metni yalnız soruyu cevaplamak için veri olarak "
                      "kullan; cevapta hesabın anlık üretildiğini ve doğrulanmadığını belirt.")

# agent_v4'ten itibaren prompt'taki "Oturum durumu" satırı (eski sürümlerde yer tutucu yok, format onu yok sayar).
TRANSCRIPT_LOADED = ("kullanıcının tarihsel transkripti bu oturumda yüklü; analyze_transcript, simulate_courses ve "
                     "confirm_equivalence onu okur.")
TRANSCRIPT_MISSING = ("bu oturumda transkript yüklü değil; kişisel durum sorularında UBYS'den indirilen tarihsel transkripti "
                      "yüklemesini iste.")
NO_TRANSCRIPT = ("Bu oturumda yüklenmiş bir transkript yok. Kullanıcıdan UBYS'den indirdiği tarihsel transkripti "
                 "yüklemesini iste.")
TURN_LIMIT_ANSWER = "Bu soruyu belirlenen adım sayısı içinde cevaplayamadım. Soruyu daha dar sorabilir misiniz?"


# Kural dosyasındaki belge adları → corpus'taki PDF: tool çıktısındaki "[<belge>, Madde N ...]" atıfları maddenin
# chunk'ına bağlanır, arayüzde tıklanınca madde açılır (#17). "Geçici Madde" ayrı chunk değil; bağlanmaz.
RULE_DOCUMENTS = {"BAİBÜ Önlisans, Lisans Eğitim-Öğretim Yönetmeliği": "önlisans lisans eğitim öğretim.pdf",
                  "BAİBÜ Alan Dışı Seçmeli Ders Yönergesi": "alan dışı seçmeli ders yönergesi.pdf"}
_RULE_CITATION = re.compile("(" + "|".join(map(re.escape, RULE_DOCUMENTS)) + r"), Madde (\d+)")
_PDF_REFERENCE = re.compile(r"\(data/([^()]+?\.pdf)\)")  # "Öğretim Planı, 2024-2025 (data/öğretim planı.pdf)"


def external_documents(plan, schedule_data):
    """Corpus dışında olup tool'ların kaynak gösterdiği belgeler: {dosya adı: başlık} (öğretim planı, ders programı).
    Başlık kaynak metninin kendisi; cevaptaki atıf ("[... (data/x.pdf)]") bu başlıkla eşleşir."""
    out = {}
    for text in (plan.get("kaynak", ""), schedule_data.get("kaynak", "")):
        m = _PDF_REFERENCE.search(text)
        if m:
            out[m.group(1)] = text
    return out


def _same_name(a, b):
    return graduation.name_similarity(a, b) == 1.0


def make_tools(retriever, plan, transcript, confirmations, today, with_requirements=False, transcript_detail=False,
               code_runner=None, generated=None, found=None):
    """Bu çağrıya (oturuma) bağlı tool'lar. `transcript` ve `confirmations` closure'da durur; LLM'in görebileceği
    argümanlar değildir. confirm_equivalence `confirmations` sözlüğünü yerinde günceller (çağıran taraf saklar)."""

    def search_regulations(query: str) -> str:
        """Üniversitenin yönetmelik ve yönergelerinde (staj, ders kaydı, sınavlar, mezuniyet, yatay geçiş, lisansüstü vb.)
        arama yapar ve en ilgili parçaları kaynak etiketleriyle döndürür.

        Args:
            query: Aranacak konu, Türkçe ve bağımsız bir soru ya da ifade olarak (ör. "staj başvurusu nasıl yapılır").
        """
        docs = retriever.search(query)
        if found is not None:  # cevabın kaynakları (arayüzde düğme ve atıf linki)
            found.extend(docs)
        return _format_context(docs)

    def analyze_transcript() -> str:
        """Oturumda yüklü transkripti öğretim planıyla karşılaştırır: şu an geçilen AKTS, bu dönemin dersleri
        geçilirse ulaşılacak AKTS ve eksik zorunlu dersler (durumlarıyla). Transkript yüklenmemişse bunu bildirir."""
        if transcript is None:
            return NO_TRANSCRIPT
        return graduation.summary_text(graduation.evaluate(transcript, plan, confirmations=confirmations), plan,
                                       enrollment=transcript_detail)

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

    def simulate_courses(course_sets: list[list[str]]) -> str:
        """What-if: oturumdaki transkripte varsayımsal dersler ekleyip mezuniyet şartlarını yeniden hesaplar. Her küme
        ayrı hesaplanır (bu dönemin dersleri ve kümedeki dersler geçilirse); hangi kümenin şartları tamamladığını,
        hangisinin tamamlamadığını ve eksik kalanları döndürür. Yalnız AKTS ve ders şartları hesaplanır; GNO
        transkriptteki güncel değerdir. Bu dönem açılmayan seçmelileri işaretler, açık yuvalar için bu dönem açılan
        dersleri listeler.

        Args:
            course_sets: Karşılaştırılacak ders kümeleri; her küme öğretim planındaki ders adlarının listesi
                (ör. [["Yapay Zeka", "Siber Güvenlik"], ["Görüntü İşleme"]]). Tek seçenek için tek küme.
        """
        if transcript is None:
            return NO_TRANSCRIPT
        try:
            reports = [graduation.simulate(transcript, plan, names, confirmations=confirmations)
                       for names in course_sets]
        except graduation.UnknownCourseError as e:
            return (f"{e}. Dersleri öğretim planındaki adlarıyla ver (analiz raporundaki eksik dersler ya da seçmeli "
                    "havuzlarındaki adlar); ad belirsizse kullanıcıya hangi dersi kastettiğini sor.")
        return graduation.simulation_text(graduation.evaluate(transcript, plan, confirmations=confirmations),
                                          reports, plan, offered=schedule.offered_electives(schedule.load_schedule(),
                                                                                            plan))

    def get_offered_electives(yuva: str = "") -> str:
        """Bilgisayar Mühendisliği bu dönemki ders programından seçmeli yuvalarında bu dönem açılan dersleri (gün, saat,
        şube, derslik) ve havuzda olup bu dönem açılmayanları döndürür. Rektörlük ve alan dışı seçmelilerini başka
        birimler açtığı için onlar için "ilgili birimden teyit et" der.

        Args:
            yuva: Seçmeli yuvası (ör. "Seçmeli III", "Seçmeli VII", "Rek. Seçmeli I", "Alan Dışı"); bütün yuvalar
                için boş.
        """
        offered = schedule.offered_electives(schedule.load_schedule(), plan)
        if not yuva.strip():
            return schedule.offered_electives_text(offered)
        try:
            slot = schedule.find_slot(yuva, plan)
        except ValueError as e:
            return str(e)
        return schedule.offered_electives_text(offered, slot["ad"])

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

    def get_graduation_requirements() -> str:
        """Bölümün mezuniyet şartlarını transkript olmadan döndürür: öğretim planının toplam AKTS'si, zorunlu dersler,
        stajlar, seçmeli yuvaları ve AKTS'leri, Alan Dışı Seçmeli Ders ve kayıt yılı istisnası, GNO alt sınırı ve
        geçer notlar; her kural kaynağıyla (belge ve madde)."""
        text = graduation.requirements_text(graduation.graduation_requirements(plan))
        if transcript_detail and transcript is not None:
            text += ("\n\nBu oturumda transkript yüklü. Soran kendi durumunu soruyorsa (ör. bu şart bana uygulanıyor mu, "
                     "eksiğim var mı) analyze_transcript tool'unu çağır ve cevabı kayıt yılına ve rapora göre ver.")
        return text

    def request_new_tool(gorev: str, transkript_gerekli: bool = False) -> str:
        """Mevcut tool'ların hiçbiri soruyu cevaplayamadığında kullanılır: görev için anlık Python kodu yazılır ve izole
        bir sandbox'ta çalıştırılır. Sandbox'ta öğretim planı (dersler, AKTS, yarıyıllar, seçmeli yuvaları ve havuzları)
        ve bu dönemin ders programı (gün, saat, sınıf, şube, derslik) bulunur; transkript_gerekli true ise oturumdaki
        transkript de (dersler, notlar, dönemler, kayıt yılı) eklenir.

        Args:
            gorev: Hesaplanacak şey; bağımsız ve açık bir Türkçe görev (ör. "Bu dönemin ders programında 3. sınıfın
                haftalık toplam ders saati").
            transkript_gerekli: Görev kullanıcının transkriptindeki verilere dayanıyorsa true.
        """
        files = {"ogretim_plani.json": json.dumps(plan, ensure_ascii=False),
                 "ders_programi.json": json.dumps(schedule.load_schedule(), ensure_ascii=False)}
        if transkript_gerekli:
            if transcript is None:
                return NO_TRANSCRIPT
            files["transkript.json"] = json.dumps(transcript, ensure_ascii=False)
        # Log kaydı zinciri ayırır: görevi Gemini yazdı, adımları ve cevabı Haiku üretti, son cümleyi yine Gemini kurar.
        record = {"gorev": gorev, "transkript": transkript_gerekli}
        generated.append(record)
        try:
            result = code_runner.run(gorev, files)
        except Exception as e:  # agent döngüsü hatayı LLM'e verir; log'da da hangi aşamada olduğu görünsün
            record["hata"] = f"{type(e).__name__}: {str(e)[:300]}"
            raise
        record.update(haiku_cevabi=result["answer"], adimlar=result["steps"], istek_sayisi=result["requests"],
                      sure_ms=result["duration_ms"], tokens=result["tokens"], bitis=result["stop_reason"])
        return f"{CODE_RESULT_HEADER}\n\n{result['answer']}"

    tools = (search_regulations, analyze_transcript, confirm_equivalence, simulate_courses, get_offered_electives,
             get_course_schedule) + ((get_graduation_requirements,) if with_requirements else ()) + (
             (request_new_tool,) if code_runner is not None else ())
    return [StructuredTool.from_function(f) for f in tools]


class AdvisorAgent:
    """Stateless: geçmişi ve transkripti çağıran taraf tutar (chat.py'de değişkenler, ileride API'de istemci)."""

    def __init__(self, retriever, llm, plan, prompt_version=AGENT_PROMPT_VERSION, code_runner=None):
        self.retriever = retriever
        self.llm = llm
        self.plan = plan
        self.system = AGENT_PROMPTS[prompt_version]
        self.with_requirements = prompt_version in REQUIREMENTS_TOOL_PROMPTS
        self.transcript_detail = prompt_version in TRANSCRIPT_DETAIL_PROMPTS
        # Anlık tool yalnız v7'de ve bir çalıştırıcı verildiyse açılır (testler sahtesini verir).
        self.code_runner = code_runner if prompt_version in CODE_TOOL_PROMPTS else None
        self.documents = external_documents(plan, schedule.load_schedule())
        # (belge, madde) -> maddenin ilk chunk'ı: kural atıflarını kaynağa çevirmek için.
        self._articles = {}
        for c in getattr(retriever, "chunks", None) or []:
            if c.metadata.get("madde") is not None:
                self._articles.setdefault((c.metadata["source"], c.metadata["madde"]), c)

    @classmethod
    def from_config(cls, cfg, prompt_version=AGENT_PROMPT_VERSION):
        """Production RagBot'un retrieval hattı ve modeliyle (aynı config) kurulur."""
        store, chunks, _ = ingest.load_index(cfg)
        # timeout: tool sonucu içeren isteklerde API bazen dakikalarca cevap vermiyor (2026-10-21); süresiz beklemek
        # yerine hata verilir, chat.py bunu yakalayıp sohbete devam eder.
        llm = ChatGoogleGenerativeAI(model=AGENT_LLM_MODEL, temperature=cfg.llm_temperature, max_retries=2,
                                     timeout=REQUEST_TIMEOUT_S)
        import codeact  # anthropic paketi yalnız anlık tool'u kullanan sürümde gerekir
        return cls(Retriever(cfg, store, chunks), llm, graduation.load_plan(), prompt_version=prompt_version,
                   code_runner=codeact.CodeRunner() if prompt_version in CODE_TOOL_PROMPTS else None)

    def ask(self, question, history=None, transcript=None, on_tool_call=None, confirmations=None):
        """Cevap üretir. `on_tool_call(name, args)` verilirse her tool çağrısında çağrılır (terminalde göstermek için).
        `confirmations` oturumun eşdeğerlik teyitleridir ({plan dışı ders kodu: plan ders kodu | None}); transkript
        gibi çağıran taraf saklar, confirm_equivalence yerinde günceller.

        Dönüş: {"answer", "tool_calls": [{"name", "args"}], "turns", "usage"}.
        """
        today = date.today()
        generated = []  # request_new_tool'un ürettiği kodlar (log için)
        found = []      # cevabın dayandığı chunk'lar: search_regulations sonuçları ve tool çıktılarındaki kural atıfları
        documents = {}  # corpus dışı kaynak belgeler (öğretim planı, ders programı)
        tools = make_tools(self.retriever, self.plan, transcript, {} if confirmations is None else confirmations, today,
                           with_requirements=self.with_requirements, transcript_detail=self.transcript_detail,
                           code_runner=self.code_runner, generated=generated, found=found)
        by_name = {t.name: t for t in tools}
        llm = self.llm.bind_tools(tools)
        system = self.system.format(no_answer=config.NO_ANSWER, today=today.strftime("%d.%m.%Y"),
                                    transcript_state=TRANSCRIPT_LOADED if transcript else TRANSCRIPT_MISSING)
        messages = [SystemMessage(system), *_to_messages(history), HumanMessage(question)]

        calls, usage = [], []
        for turn in range(1, MAX_TURNS + 1):
            message = llm.invoke(messages)
            usage.append(message.usage_metadata)
            messages.append(message)
            if not message.tool_calls:
                return {"answer": message.text.strip(), "tool_calls": calls, "turns": turn,
                        "usage": token_usage(usage), "generated_code": generated,
                        "sources": _unique_chunks(found), "documents": list(documents.values())}
            for call in message.tool_calls:
                calls.append({"name": call["name"], "args": call["args"]})
                if on_tool_call:
                    on_tool_call(call["name"], call["args"])
                tool = by_name.get(call["name"])
                try:
                    result = tool.invoke(call["args"]) if tool else f"Bilinmeyen tool: {call['name']}"
                except Exception as e:  # hatalı argüman vb.: LLM düzeltebilsin diye hata metni geri verilir
                    result = f"Tool hatası: {type(e).__name__}: {e}"
                self._collect_cited(result, found, documents)
                messages.append(ToolMessage(content=result, tool_call_id=call["id"], name=call["name"]))
        return {"answer": TURN_LIMIT_ANSWER, "tool_calls": calls, "turns": MAX_TURNS, "usage": token_usage(usage),
                "generated_code": generated, "sources": _unique_chunks(found), "documents": list(documents.values())}

    def _collect_cited(self, text, found, documents):
        """Tool çıktısındaki kural atıflarını (maddenin ilk chunk'ı) ve corpus dışı belge referanslarını toplar."""
        for name, madde in _RULE_CITATION.findall(text):
            chunk = self._articles.get((RULE_DOCUMENTS[name], int(madde)))
            if chunk is not None:
                found.append(chunk)
        for filename in _PDF_REFERENCE.findall(text):
            if filename in self.documents:
                documents[filename] = {"source": filename, "title": self.documents[filename]}


def _unique_chunks(docs):
    seen, out = set(), []
    for d in docs:
        if d.metadata["chunk_id"] not in seen:
            seen.add(d.metadata["chunk_id"])
            out.append(d)
    return out
