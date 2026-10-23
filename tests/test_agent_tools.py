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
        self.tools = [t.name for t in tools]
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

    assert AGENT_PROMPT_VERSION == "agent_v4"  # EXP-003: transkript sorularında cevaplar düzeldi
    assert "simulate_courses" in AGENT_PROMPTS[AGENT_PROMPT_VERSION] and "get_offered_electives" in AGENT_PROMPTS[AGENT_PROMPT_VERSION]
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


def requirements_reply(prompt_version):
    llm = ScriptedChatModel([
        AIMessage("", tool_calls=[{"name": "get_graduation_requirements", "args": {}, "id": "1"}]),
        AIMessage("nihai cevap"),
    ])
    AdvisorAgent(retriever=None, llm=llm, plan=graduation.load_plan(), prompt_version=prompt_version).ask(
        "Mezun olmak için hangi şartlar gerekiyor?")
    return llm, next(m for m in llm.received[-1] if isinstance(m, ToolMessage)).content


def test_graduation_requirements_tool_works_without_a_transcript():
    llm, content = requirements_reply("agent_v5")
    assert "get_graduation_requirements" in llm.tools
    assert content == graduation.requirements_text(graduation.graduation_requirements())


def test_older_prompts_do_not_see_the_graduation_requirements_tool():
    # Eski sürümlerin deneyleri (EXP-002, EXP-003) aynı tool listesiyle tekrar üretilebilsin.
    llm, content = requirements_reply("agent_v4")
    assert "get_graduation_requirements" not in llm.tools
    assert content == "Bilinmeyen tool: get_graduation_requirements"


def test_agent_v5_prompt_adds_only_the_requirements_rule():
    from agent import AGENT_PROMPTS
    v4, v5 = AGENT_PROMPTS["agent_v4"].splitlines(), AGENT_PROMPTS["agent_v5"].splitlines()
    added = [line for line in v5 if line not in v4]
    assert len(added) == 1 and "get_graduation_requirements" in added[0]
    assert [line for line in v4 if line not in v5] == []


def tool_reply(prompt_version, name, transcript):
    llm = ScriptedChatModel([AIMessage("", tool_calls=[{"name": name, "args": {}, "id": "1"}]), AIMessage("cevap")])
    AdvisorAgent(retriever=None, llm=llm, plan=graduation.load_plan(), prompt_version=prompt_version).ask(
        "Alan dışı seçmeli ders almam gerekiyor mu?", transcript=transcript)
    return next(m for m in llm.received[-1] if isinstance(m, ToolMessage)).content


def test_agent_v6_states_enrollment_year_and_redirects_personal_questions():
    from agent import AGENT_PROMPTS
    assert AGENT_PROMPTS["agent_v6"] == AGENT_PROMPTS["agent_v5"]           # fark yalnız tool çıktılarında
    assert "Kayıt yılı: 2023-2024" in tool_reply("agent_v6", "analyze_transcript", SAMPLE)
    assert "analyze_transcript tool'unu çağır" in tool_reply("agent_v6", "get_graduation_requirements", SAMPLE)
    assert "analyze_transcript tool'unu çağır" not in tool_reply("agent_v6", "get_graduation_requirements", None)


def test_agent_v5_tool_outputs_are_unchanged():
    # EXP-004 aynı tool çıktılarıyla tekrar üretilebilsin.
    assert "Kayıt yılı" not in tool_reply("agent_v5", "analyze_transcript", SAMPLE)
    assert "analyze_transcript tool'unu çağır" not in tool_reply("agent_v5", "get_graduation_requirements", SAMPLE)


class FakeRunner:
    """CodeRunner yerine: gelen görevi ve dosyaları kaydeder, hazır sonuç döndürür."""

    def __init__(self):
        self.calls = []

    def run(self, task, files):
        self.calls.append((task, files))
        return {"answer": "Toplam 12 saat.", "steps": [{"tip": "kod", "kod": "print(12)", "cikti": "12\n"}], "requests": 1, "duration_ms": 5, "tokens": {"input": 10, "output": 2}, "stop_reason": "end_turn"}


def new_tool_reply(prompt_version, args, transcript, runner):
    llm = ScriptedChatModel([AIMessage("", tool_calls=[{"name": "request_new_tool", "args": args, "id": "1"}]),
                             AIMessage("cevap")])
    result = AdvisorAgent(retriever=None, llm=llm, plan=graduation.load_plan(), prompt_version=prompt_version,
                          code_runner=runner).ask("3. sınıfın haftada kaç saat dersi var?", transcript=transcript)
    return llm, next(m for m in llm.received[-1] if isinstance(m, ToolMessage)).content, result


def test_new_tool_sends_plan_and_schedule_but_not_the_transcript_by_default():
    runner = FakeRunner()
    llm, content, result = new_tool_reply("agent_v7", {"gorev": "3. sınıfın haftalık ders saati"}, SAMPLE, runner)
    task, files = runner.calls[0]
    assert task == "3. sınıfın haftalık ders saati"
    assert sorted(files) == ["ders_programi.json", "ogretim_plani.json"]
    assert "Toplam 12 saat." in content and "doğrulanmadı" in content
    assert result["generated_code"] == [{"gorev": task, "transkript": False, "haiku_cevabi": "Toplam 12 saat.",
                                         "adimlar": [{"tip": "kod", "kod": "print(12)", "cikti": "12\n"}], "istek_sayisi": 1, "sure_ms": 5,
                                         "tokens": {"input": 10, "output": 2}, "bitis": "end_turn"}]


def test_new_tool_sends_the_transcript_only_when_needed_and_loaded():
    runner = FakeRunner()
    new_tool_reply("agent_v7", {"gorev": "AA notlu derslerin AKTS toplamı", "transkript_gerekli": True}, SAMPLE, runner)
    files = runner.calls[0][1]
    assert json.loads(files["transkript.json"]) == SAMPLE
    _, content, _ = new_tool_reply("agent_v7", {"gorev": "AA notlu derslerin AKTS toplamı",
                                                "transkript_gerekli": True}, None, runner)
    assert content == NO_TRANSCRIPT and len(runner.calls) == 1             # transkript yoksa sandbox'a hiç gidilmez


def test_only_agent_v7_sees_the_new_tool():
    for version in ("agent_v4", "agent_v5", "agent_v6"):
        llm, content, _ = new_tool_reply(version, {"gorev": "x"}, None, FakeRunner())
        assert "request_new_tool" not in llm.tools and content == "Bilinmeyen tool: request_new_tool"
    llm, _, _ = new_tool_reply("agent_v7", {"gorev": "x"}, None, FakeRunner())
    assert "request_new_tool" in llm.tools


def test_agent_v7_prompt_adds_only_the_new_tool_rule():
    from agent import AGENT_PROMPTS
    v6, v7 = AGENT_PROMPTS["agent_v6"].splitlines(), AGENT_PROMPTS["agent_v7"].splitlines()
    added = [line for line in v7 if line not in v6]
    assert len(added) == 1 and "request_new_tool" in added[0]
    assert [line for line in v6 if line not in v7] == []


class FakeHandoffRunner:
    """CodeRunner.handoff yerine: devredilen konuşmayı, tool'ları ve dosyaları kaydeder; Haiku bir tool çağırmış gibi
    execute'u bir kez kullanır."""

    def __init__(self):
        self.received = None

    def handoff(self, system, messages, tools, execute, files):
        self.received = {"system": system, "messages": list(messages), "tools": [t.name for t in tools], "files": files}
        execute("analyze_transcript", {})
        return {"answer": "Haiku cevabı", "steps": [{"tip": "metin", "metin": "Haiku cevabı"}], "requests": 2,
                "duration_ms": 7, "tokens": {"input": 30, "output": 5}, "stop_reason": "end_turn", "tool_calls": []}


def handoff_reply(transcript):
    runner = FakeHandoffRunner()
    llm = ScriptedChatModel([AIMessage("", tool_calls=[{"name": "request_new_tool", "args": {"gorev": "ortalama"},
                                                        "id": "1"}])])
    result = AdvisorAgent(retriever=None, llm=llm, plan=graduation.load_plan(), prompt_version="agent_v9",
                          code_runner=runner).ask("62 aldım, ortalamam kaç olur?", transcript=transcript)
    return llm, runner, result


def test_agent_v9_hands_the_whole_conversation_to_the_code_model():
    from agent import HANDOFF_NOTE
    llm, runner, result = handoff_reply(SAMPLE)
    assert len(llm.received) == 1                                   # devirden sonra Gemini çağrılmaz
    got = runner.received
    assert got["system"].endswith(HANDOFF_NOTE)
    assert isinstance(got["messages"][-1], ToolMessage) and "devredildi" in got["messages"][-1].content
    assert "request_new_tool" not in got["tools"] and "analyze_transcript" in got["tools"]
    assert json.loads(got["files"]["transkript.json"]) == SAMPLE
    assert result["answer"] == "Haiku cevabı"
    assert [c["name"] for c in result["tool_calls"]] == ["request_new_tool", "analyze_transcript"]
    [record] = result["generated_code"]
    assert record["devir"] is True and record["haiku_cevabi"] == "Haiku cevabı" and record["transkript"] is True


def test_handoff_without_a_transcript_sends_no_transcript_file():
    _, runner, _ = handoff_reply(None)
    assert sorted(runner.received["files"]) == ["ders_programi.json", "ogretim_plani.json"]


def test_agent_v9_prompt_replaces_only_the_new_tool_rule():
    from agent import AGENT_PROMPTS
    v7, v9 = AGENT_PROMPTS["agent_v7"].splitlines(), AGENT_PROMPTS["agent_v9"].splitlines()
    assert [line for line in v9 if line not in v7] == [next(l for l in v9 if "devredilir" in l)]


# --- Kaynaklar (#16, #17): cevabın dayandığı chunk'lar ve corpus dışı belgeler ---

from langchain_core.documents import Document  # noqa: E402


def chunk(chunk_id, source, madde):
    return Document(page_content=f"MADDE {madde}- ...", metadata={"chunk_id": chunk_id, "source": source,
                                                                   "title": source[:-4], "page": 1, "madde": madde})


CHUNKS = [chunk(1, "önlisans lisans eğitim öğretim.pdf", 35), chunk(2, "önlisans lisans eğitim öğretim.pdf", 35),
          chunk(3, "alan dışı seçmeli ders yönergesi.pdf", 1), chunk(4, "alan dışı seçmeli ders yönergesi.pdf", 7),
          chunk(5, "mühendislik fakültesi staj yönergesi.pdf", 10)]


class FakeRetriever:
    chunks = CHUNKS

    def search(self, query):
        return [CHUNKS[4]]


def sources_of(calls, transcript=None, plan=None):
    llm = ScriptedChatModel([AIMessage("", tool_calls=calls), AIMessage("cevap")])
    return AdvisorAgent(retriever=FakeRetriever(), llm=llm, plan=plan or graduation.load_plan(),
                        prompt_version="agent_v6").ask("soru", transcript=transcript)


def test_search_results_become_sources():
    result = sources_of([{"name": "search_regulations", "args": {"query": "staj"}, "id": "1"}])
    assert [d.metadata["chunk_id"] for d in result["sources"]] == [5]


def test_rule_citations_in_tool_output_link_to_the_first_chunk_of_the_article():
    result = sources_of([{"name": "get_graduation_requirements", "args": {}, "id": "1"}])
    ids = [d.metadata["chunk_id"] for d in result["sources"]]
    assert 1 in ids and 2 not in ids and 4 in ids        # Madde 35'in ilk chunk'ı, ADSD Madde 7
    assert 3 not in ids                                    # "Geçici Madde 1" Madde 1'e bağlanmaz
    assert len(ids) == len(set(ids))


def test_plan_and_schedule_pdfs_become_document_sources():
    result = sources_of([{"name": "analyze_transcript", "args": {}, "id": "1"},
                         {"name": "get_offered_electives", "args": {"yuva": "Seçmeli III"}, "id": "2"}],
                        transcript=SAMPLE)
    assert {d["source"] for d in result["documents"]} == {"öğretim planı.pdf", "2026_27_Güz_Lisans_v2.pdf"}
    assert all(d["title"].endswith(f"(data/{d['source']})") for d in result["documents"])
