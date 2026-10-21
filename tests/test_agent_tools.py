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
