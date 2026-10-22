"""API: transkript yükleme ve agent sohbeti; gerçek modeller ve LLM yok.

ChatService gerçek (log yazımı dahil), ama içindeki RagBot sahte ve agent'ın LLM'i hazır mesajlar döndürür;
log'lar geçici bir klasöre yazılır. Uygulama app.dependency_overrides ile bu service'i kullanır. TestClient
`with` olmadan kullanılır: lifespan (gerçek modellerin yüklenmesi) çalışmaz.
Gerçek transkriptli testler yalnız yerelde çalışır (tests/private/, git'e girmez).
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import transcript
from agent import AdvisorAgent
from app.api import app, get_service
from app.service import ChatService

PRIVATE = Path(__file__).parent / "private"
PERSONAL_FILE = PRIVATE / "kisisel.json"
PERSONAL = json.loads(PERSONAL_FILE.read_text(encoding="utf-8")) if PERSONAL_FILE.exists() else {}
HISTORICAL = PRIVATE / PERSONAL.get("tarihsel_transkript", "-")
SEMESTER_LAYOUT = PRIVATE / PERSONAL.get("yariyil_transkript", "-")

PLAN = {
    "bolum": "Bilgisayar Mühendisliği", "katalog_yili": "2024-2025", "kaynak": "test planı",
    "zorunlu_dersler": [
        {"kod": "Z1", "ad": "Matematik I", "yariyil": 1, "kredi": 4, "akts": 6},
        {"kod": "MIK", "ad": "Mikroişlemciler", "ad_en": "Microprocessors", "yariyil": 5, "kredi": 4, "akts": 5},
    ],
}
SAMPLE = {
    "bolum": "Bilgisayar Mühendisliği", "egitim_duzeyi": "Lisans", "kayit_yili": 2023, "kayit_donemi": "Güz",
    "dersler": [
        {"yil": 2023, "donem": "Güz", "kod": "Z1", "ad": "Matematik I", "kredi": 4, "akts": 6, "not": "CC",
         "erasmus": False},
        {"yil": 2025, "donem": "Güz", "kod": "E2", "ad": "Microcomputer Systems", "kredi": 3, "akts": 5,
         "not": "AA", "erasmus": True},
    ],
    "obs_ozet": {"birikimli_alinan_akts": 11, "birikimli_tamamlanan_akts": 11, "gno": 3.1},
}
QUESTION = "Mezun olmak için neyim eksik?"
ANSWER = "Transkriptteki öğrencinin GNO'su 3,1; Mikroişlemciler eşdeğer sayıldı."


class ScriptedChatModel:
    """Sırayla hazır mesajlar döndürür (bind_tools kendini döndürür)."""

    def __init__(self, replies):
        self.replies = list(replies)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self.replies.pop(0)


class FakeBot:
    """RagBot yerine: chunk listesi boş, ask sabit bir cevap döner."""
    retriever = SimpleNamespace(chunks=[])

    def ask(self, question, history=None):
        return {"answer": "mevzuat cevabı", "query": question, "docs": []}


def make_client(tmp_path, replies=(), **agent_kwargs):
    agent = AdvisorAgent(retriever=None, llm=ScriptedChatModel(replies), plan=PLAN, **agent_kwargs)
    service = ChatService(FakeBot(), "EXP-TEST", log_dir=tmp_path, agent=agent)
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def upload(client, data):
    return client.post("/transcript", content=data, headers={"Content-Type": "application/pdf"})


def log_text(log_dir):
    return "\n".join(p.read_text(encoding="utf-8") for p in Path(log_dir).iterdir() if p.is_file())


# --- POST /transcript ---

def test_upload_returns_parsed_transcript_and_writes_nothing(tmp_path, monkeypatch):
    received = []

    def fake_parse(source):
        received.append(source.read())  # dosya yolu değil, bellekteki dosya nesnesi
        return SAMPLE

    monkeypatch.setattr(transcript, "parse_transcript", fake_parse)
    client = make_client(tmp_path)
    res = upload(client, b"%PDF-1.4 sahte")
    assert res.status_code == 200
    assert res.json() == {"transcript": SAMPLE, "course_count": 2}
    assert received == [b"%PDF-1.4 sahte"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("data", [b"bu bir PDF degil", b""])
def test_unreadable_file_asks_for_historical_transcript(tmp_path, data):
    res = upload(make_client(tmp_path), data)
    assert res.status_code == 422
    assert transcript.ASK_FOR_HISTORICAL in res.json()["detail"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(not SEMESTER_LAYOUT.is_file(), reason="gerçek yarıyıl transkripti yok")
def test_semester_layout_transcript_is_rejected(tmp_path):
    res = upload(make_client(tmp_path), SEMESTER_LAYOUT.read_bytes())
    assert res.status_code == 422
    assert "yarıyıl düzenli" in res.json()["detail"]
    assert transcript.ASK_FOR_HISTORICAL in res.json()["detail"]


@pytest.mark.skipif(not HISTORICAL.is_file(), reason="gerçek tarihsel transkript yok")
def test_real_historical_transcript_upload_has_no_personal_values(tmp_path):
    res = upload(make_client(tmp_path), HISTORICAL.read_bytes())
    assert res.status_code == 200
    body = res.json()
    assert body["course_count"] == len(body["transcript"]["dersler"]) > 0
    for value in PERSONAL["kisisel_degerler"]:
        assert value not in res.text
    assert list(tmp_path.iterdir()) == []


# --- POST /agent/ask ---

def confirm_then_answer():
    return [
        AIMessage("", tool_calls=[{"name": "confirm_equivalence", "id": "1",
                                   "args": {"transcript_course": "Microcomputer Systems",
                                            "plan_course": "Mikroişlemciler"}}]),
        AIMessage(ANSWER),
    ]


def test_agent_ask_uses_transcript_and_returns_updated_confirmations(tmp_path):
    client = make_client(tmp_path, confirm_then_answer())
    res = client.post("/agent/ask", json={"question": QUESTION, "transcript": SAMPLE, "confirmations": {}})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == ANSWER
    assert body["tool_calls"] == [{"name": "confirm_equivalence",
                                   "args": {"transcript_course": "Microcomputer Systems",
                                            "plan_course": "Mikroişlemciler"}}]
    assert body["confirmations"] == {"E2": "MIK"}


def test_agent_log_keeps_question_answer_and_tool_calls(tmp_path):
    client = make_client(tmp_path, confirm_then_answer())
    client.post("/agent/ask", json={"question": QUESTION, "transcript": SAMPLE,
                                    "history": [{"role": "user", "content": "Önceki soru"}]})
    assert [p.name for p in tmp_path.iterdir()] == ["agent.jsonl"]
    record = json.loads(log_text(tmp_path))
    assert (record["soru"], record["cevap"]) == (QUESTION, ANSWER)
    assert record["tool_cagrilari"] == [{"name": "confirm_equivalence",
                                         "args": {"transcript_course": "Microcomputer Systems",
                                                  "plan_course": "Mikroişlemciler"}}]
    assert record["transkript"] is True
    assert "Önceki soru" not in log_text(tmp_path)                      # geçmiş ve transkript yazılmaz
    assert record["hata"] is None and record["latency_ms"] is not None


def test_agent_ask_without_transcript_reaches_tools_and_is_logged(tmp_path):
    # Web sohbeti transkript yokken de agent'a gider (#13): ders programı gibi tool'lar transkriptsiz çalışır.
    client = make_client(tmp_path, [
        AIMessage("", tool_calls=[{"name": "get_course_schedule", "id": "1", "args": {"gun": "Pazartesi", "sinif": 2}}]),
        AIMessage(ANSWER),
    ])
    res = client.post("/agent/ask", json={"question": "Pazartesi 2. sınıfın dersleri?"})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == ANSWER
    assert [c["name"] for c in body["tool_calls"]] == ["get_course_schedule"]
    record = json.loads(log_text(tmp_path))
    assert record["transkript"] is False and record["tool_lar"] == ["get_course_schedule"]


class FakeCodeRunner:
    def run(self, task, files):
        return {"answer": "Toplam 21 saat.", "steps": [{"tip": "kod", "kod": "print(21)", "cikti": "21\n"}], "requests": 1, "duration_ms": 5,
                "tokens": {"input": 10, "output": 2}, "stop_reason": "end_turn"}


def test_generated_code_is_logged(tmp_path):
    # Anlık tool'un yazdığı kod log'da (#18): hangi hesapların kalıcı tool olması gerektiği buradan görülür.
    client = make_client(tmp_path, [
        AIMessage("", tool_calls=[{"name": "request_new_tool", "id": "1", "args": {"gorev": "haftalık ders saati"}}]),
        AIMessage(ANSWER),
    ], prompt_version="agent_v7", code_runner=FakeCodeRunner())
    assert client.post("/agent/ask", json={"question": "3. sınıfın haftada kaç saat dersi var?"}).status_code == 200
    record = json.loads(log_text(tmp_path))
    [item] = record["uretilen_kod"]
    assert (item["gorev"], item["haiku_cevabi"]) == ("haftalık ders saati", "Toplam 21 saat.")
    assert item["adimlar"] == [{"tip": "kod", "kod": "print(21)", "cikti": "21\n"}]
    assert record["cevap"] == ANSWER                        # Gemini'nin son cümlesi ayrı: kim neyi yazdı görünür


class FailingCodeRunner:
    def run(self, task, files):
        raise TimeoutError("sandbox cevap vermedi")


def test_failed_code_run_is_logged_with_its_error(tmp_path):
    client = make_client(tmp_path, [
        AIMessage("", tool_calls=[{"name": "request_new_tool", "id": "1", "args": {"gorev": "haftalık ders saati"}}]),
        AIMessage(ANSWER),
    ], prompt_version="agent_v7", code_runner=FailingCodeRunner())
    client.post("/agent/ask", json={"question": "3. sınıfın haftada kaç saat dersi var?"})
    record = json.loads(log_text(tmp_path))
    assert record["uretilen_kod"] == [{"gorev": "haftalık ders saati", "transkript": False,
                                       "hata": "TimeoutError: sandbox cevap vermedi"}]


def test_agent_error_is_logged_by_type_only(tmp_path):
    client = make_client(tmp_path, [])  # hazır mesaj yok: LLM çağrısı IndexError fırlatır
    res = client.post("/agent/ask", json={"question": QUESTION, "transcript": SAMPLE})
    assert res.status_code == 500
    record = json.loads(log_text(tmp_path))
    assert record["hata"] == "IndexError"


@pytest.mark.skipif(not HISTORICAL.is_file(), reason="gerçek tarihsel transkript yok")
def test_real_transcript_flow_leaves_no_personal_or_transcript_data_in_logs(tmp_path):
    client = make_client(tmp_path, [
        AIMessage("", tool_calls=[{"name": "analyze_transcript", "args": {}, "id": "1"}]),
        AIMessage(ANSWER),
    ])
    parsed = upload(client, HISTORICAL.read_bytes()).json()["transcript"]
    res = client.post("/agent/ask", json={"question": QUESTION, "transcript": parsed})
    assert res.status_code == 200
    text = log_text(tmp_path)
    for value in PERSONAL["kisisel_degerler"]:
        assert value not in text
    for course in parsed["dersler"]:
        assert course["ad"] not in text


# --- POST /ask (transkriptsiz sohbet eskisi gibi) ---

def test_classic_ask_is_unchanged(tmp_path):
    client = make_client(tmp_path)
    res = client.post("/ask", json={"question": "Devamsızlık sınırı nedir?"})
    assert res.status_code == 200
    assert res.json()["answer"] == "mevzuat cevabı"
    record = json.loads((tmp_path / "sohbet.jsonl").read_text(encoding="utf-8"))
    assert record["soru"] == "Devamsızlık sınırı nedir?"
