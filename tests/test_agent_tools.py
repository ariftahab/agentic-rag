"""Agent'ın tool bağlantısı: sahte bir chat model ile, ağ ve LLM yok.

Agent döngüsünün kalitesi (tool seçimi, cevap doğruluğu) burada değil, deney hattında ölçülür. Burada yalnız
ucuz ve gerçek iki şey korunur: analyze_transcript oturumdaki transkripti değerlendirme modülüne veriyor ve
LLM'e giden mesajlarda transkriptteki kişisel değerler bulunmuyor.
"""
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

import graduation
import schedule
from agent import AdvisorAgent, NO_TRANSCRIPT

PLAN = {
    "bolum": "Bilgisayar Mühendisliği", "katalog_yili": "2024-2025", "kaynak": "test planı",
    "zorunlu_dersler": [
        {"kod": "Z1", "ad": "Matematik I", "yariyil": 1, "kredi": 4, "akts": 6},
        {"kod": "Z2", "ad": "Fizik I", "yariyil": 1, "kredi": 4, "akts": 6},
        {"kod": "Z3", "ad": "Bitirme Projesi I", "yariyil": 7, "kredi": 3, "akts": 5},
    ],
}


def _course(year, term, code, name, ects, grade):
    return {"yil": year, "donem": term, "kod": code, "ad": name, "kredi": 3, "akts": ects, "not": grade,
            "erasmus": False}


SAMPLE = {
    "bolum": "Bilgisayar Mühendisliği", "egitim_duzeyi": "Lisans", "kayit_yili": 2023, "kayit_donemi": "Güz",
    "dersler": [_course(2023, "Güz", "Z1", "Matematik I", 6, "CC"), _course(2023, "Güz", "Z2", "Fizik I", 6, "FF"),
                _course(2026, "Güz", "Z3", "Bitirme Projesi I", 5, None)],
    "obs_ozet": {"birikimli_alinan_akts": 12, "birikimli_tamamlanan_akts": 6, "gno": 2.5},
}

PRIVATE = Path(__file__).parent / "private"
PERSONAL_FILE = PRIVATE / "kisisel.json"
PERSONAL = json.loads(PERSONAL_FILE.read_text(encoding="utf-8")) if PERSONAL_FILE.exists() else {}
HISTORICAL = PRIVATE / PERSONAL.get("tarihsel_transkript", "-")


class ScriptedChatModel:
    """Sırayla hazır mesajlar döndürür ve kendisine gönderilen mesajları kaydeder (bind_tools kendini döndürür)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.received = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.received.append(list(messages))
        return self.replies.pop(0)


def ask_with_analysis(transcript, plan):
    llm = ScriptedChatModel([
        AIMessage("", tool_calls=[{"name": "analyze_transcript", "args": {}, "id": "1"}]),
        AIMessage("nihai cevap"),
    ])
    result = AdvisorAgent(retriever=None, llm=llm, plan=plan).ask("Mezun olmak için neyim eksik?",
                                                                  transcript=transcript)
    tool_messages = [m for m in llm.received[-1] if isinstance(m, ToolMessage)]
    return result, llm, tool_messages


def test_analyze_transcript_evaluates_session_transcript():
    result, _, tool_messages = ask_with_analysis(SAMPLE, PLAN)
    assert result["answer"] == "nihai cevap"
    assert result["tool_calls"] == [{"name": "analyze_transcript", "args": {}}]
    assert tool_messages[0].content == graduation.summary_text(graduation.evaluate(SAMPLE, PLAN), PLAN)


def test_analyze_transcript_without_transcript_asks_for_historical():
    _, _, tool_messages = ask_with_analysis(None, PLAN)
    assert tool_messages[0].content == NO_TRANSCRIPT


@pytest.mark.skipif(not HISTORICAL.is_file(), reason="gerçek tarihsel transkript yok")
def test_llm_never_receives_personal_values():
    import transcript

    parsed = transcript.parse_transcript(HISTORICAL)
    _, llm, _ = ask_with_analysis(parsed, graduation.load_plan())
    sent = "\n".join(str(m.content) for messages in llm.received for m in messages)
    for value in PERSONAL["kisisel_degerler"]:
        assert value not in sent


ERASMUS_PLAN = {**PLAN, "zorunlu_dersler": PLAN["zorunlu_dersler"] + [
    {"kod": "MIK", "ad": "Mikroişlemciler", "ad_en": "Microprocessors", "yariyil": 5, "kredi": 4, "akts": 5}]}
ERASMUS_SAMPLE = {**SAMPLE, "dersler": SAMPLE["dersler"] + [
    {**_course(2025, "Güz", "E2", "Microcomputer Systems", 5, "AA"), "erasmus": True}]}


def confirm(args, confirmations):
    llm = ScriptedChatModel([
        AIMessage("", tool_calls=[{"name": "confirm_equivalence", "args": args, "id": "1"}]),
        AIMessage("nihai cevap"),
    ])
    AdvisorAgent(retriever=None, llm=llm, plan=ERASMUS_PLAN).ask(
        "Microcomputer Systems Mikroişlemciler yerine sayıldı", transcript=ERASMUS_SAMPLE, confirmations=confirmations)
    return next(m for m in llm.received[-1] if isinstance(m, ToolMessage)).content


def test_confirm_equivalence_is_kept_in_the_session_and_applied():
    confirmations = {}
    content = confirm({"transcript_course": "Microcomputer Systems", "plan_course": "Mikroişlemciler"}, confirmations)
    assert confirmations == {"E2": "MIK"}
    assert "Microcomputer Systems (5 AKTS) → Mikroişlemciler: kullanıcı teyit etti" in content


def test_confirm_equivalence_records_no_equivalent():
    confirmations = {}
    confirm({"transcript_course": "microcomputer systems", "plan_course": ""}, confirmations)
    assert confirmations == {"E2": None}


def test_confirm_equivalence_with_unknown_plan_course_is_not_recorded():
    confirmations = {}
    content = confirm({"transcript_course": "Microcomputer Systems", "plan_course": "Kuantum Fiziği"}, confirmations)
    assert confirmations == {}
    assert "Kuantum Fiziği" in content and "bulunamadı" in content


ELECTIVE_PLAN = {**PLAN, "secmeli_yuvalar": [
    {"ad": "Seçmeli VII", "kod": "S7", "yariyil": 7, "akts": 8, "tur": "bolum",
     "havuz": [{"kod": "YZ", "ad": "Yapay Zeka", "akts": 4}, {"kod": "YSA", "ad": "Yapay Sinir Ağları", "akts": 4}]}]}


def simulate_reply(args, transcript):
    llm = ScriptedChatModel([
        AIMessage("", tool_calls=[{"name": "simulate_courses", "args": args, "id": "1"}]),
        AIMessage("nihai cevap"),
    ])
    AdvisorAgent(retriever=None, llm=llm, plan=ELECTIVE_PLAN).ask("Bu dersleri alırsam AKTS tamamlanır mı?",
                                                                  transcript=transcript)
    return next(m for m in llm.received[-1] if isinstance(m, ToolMessage)).content


def test_simulate_courses_evaluates_each_set_on_the_session_transcript():
    # Açılan seçmeliler gerçek ders programından; test planının kodları programda olmadığı için hepsi "açılmıyor".
    sets = [["Yapay Zeka", "Yapay Sinir Ağları"], ["Yapay Zeka"]]
    expected = graduation.simulation_text(
        graduation.evaluate(SAMPLE, ELECTIVE_PLAN),
        [graduation.simulate(SAMPLE, ELECTIVE_PLAN, names) for names in sets], ELECTIVE_PLAN,
        offered=schedule.offered_electives(schedule.load_schedule(), ELECTIVE_PLAN))
    content = simulate_reply({"course_sets": sets}, SAMPLE)
    assert content == expected
    assert "Yapay Sinir Ağları bu dönem açılmıyor" in content


def test_simulate_courses_with_unknown_course_asks_for_plan_names():
    content = simulate_reply({"course_sets": [["Kuantum Fiziği"]]}, SAMPLE)
    assert "Kuantum Fiziği" in content and "öğretim planındaki adlarıyla" in content


def test_simulate_courses_without_transcript_asks_for_historical():
    assert simulate_reply({"course_sets": [["Yapay Zeka"]]}, None) == NO_TRANSCRIPT


def test_default_prompt_knows_the_what_if_and_offered_electives_tools():
    from agent import AGENT_PROMPTS, AGENT_PROMPT_VERSION

    assert AGENT_PROMPT_VERSION == "agent_v3"
    assert "simulate_courses" in AGENT_PROMPTS["agent_v3"] and "get_offered_electives" in AGENT_PROMPTS["agent_v3"]
    assert "get_offered_electives" not in AGENT_PROMPTS["agent_v2"]      # eski sürümler değişmez
    assert "simulate_courses" in AGENT_PROMPTS["agent_v2"] and "UBYS" in AGENT_PROMPTS["agent_v2"]
    assert "OBS'den" in AGENT_PROMPTS["agent_v1"]


def offered_reply(args):
    llm = ScriptedChatModel([
        AIMessage("", tool_calls=[{"name": "get_offered_electives", "args": args, "id": "1"}]),
        AIMessage("nihai cevap"),
    ])
    AdvisorAgent(retriever=None, llm=llm, plan=graduation.load_plan()).ask("Seçmeli III için ne açılıyor?")
    return next(m for m in llm.received[-1] if isinstance(m, ToolMessage)).content


def test_offered_electives_for_one_slot_without_a_transcript():
    plan = graduation.load_plan()
    expected = schedule.offered_electives_text(schedule.offered_electives(schedule.load_schedule(), plan),
                                               "Seçmeli III")
    content = offered_reply({"yuva": "seçmeli 3"})
    assert content == expected and "Web Programlama" in content


def test_offered_electives_with_unknown_slot_lists_the_slot_names():
    content = offered_reply({"yuva": "Seçmeli XII"})
    assert "Seçmeli XII" in content and "Seçmeli III" in content


def schedule_reply(args, transcript):
    llm = ScriptedChatModel([
        AIMessage("", tool_calls=[{"name": "get_course_schedule", "args": args, "id": "1"}]),
        AIMessage("nihai cevap"),
    ])
    AdvisorAgent(retriever=None, llm=llm, plan=PLAN).ask("Ders programım?", transcript=transcript)
    return next(m for m in llm.received[-1] if isinstance(m, ToolMessage)).content


def test_course_schedule_without_transcript_or_filter_asks_for_class():
    assert "kaçıncı sınıf" in schedule_reply({"gun": "2026-10-22"}, None)


def test_course_schedule_uses_this_terms_courses_from_the_transcript():
    transcript = {**SAMPLE, "dersler": [_course(2026, "Güz", "1906004232018", "Bilişim Hukuku", 4, None)]}
    content = schedule_reply({"gun": "2026-10-22"}, transcript)
    assert "- 12:30-15:15 Bilişim Hukuku (Seç VII), 4. sınıf, 326" in content
    assert "bütün şubeler" in content


@pytest.mark.parametrize("transcript, state", [(SAMPLE, "TRANSCRIPT_LOADED"), (None, "TRANSCRIPT_MISSING")])
def test_agent_v4_prompt_states_whether_a_transcript_is_loaded(transcript, state):
    # EXP-002: agent oturumda transkript yüklü olduğunu bilmediği için kişisel sorularda analyze_transcript çağırmadı.
    import agent
    llm = ScriptedChatModel([AIMessage("cevap")])
    AdvisorAgent(retriever=None, llm=llm, plan=PLAN, prompt_version="agent_v4").ask("Stajlarım tamam mı?",
                                                                                   transcript=transcript)
    assert f"Oturum durumu: {getattr(agent, state)}" in llm.received[0][0].content


def test_older_prompts_have_no_session_state_line():
    from agent import AGENT_PROMPTS
    assert all("Oturum durumu" not in AGENT_PROMPTS[v] for v in ("agent_v1", "agent_v2", "agent_v3"))
