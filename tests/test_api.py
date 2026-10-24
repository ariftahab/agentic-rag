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
    agent = AdvisorAgent(llm=ScriptedChatModel(replies), plan=PLAN, **{"retriever": None, **agent_kwargs})
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
    record = json.loads((tmp_path / "agent.jsonl").read_text(encoding="utf-8"))
    [item] = record["uretilen_kod"]
    assert (item["gorev"], item["haiku_cevabi"]) == ("haftalık ders saati", "Toplam 21 saat.")
    assert item["adimlar"] == [{"tip": "kod", "kod": "print(21)", "cikti": "21\n"}]
    assert record["cevap"] == ANSWER                        # Gemini'nin son cümlesi ayrı: kim neyi yazdı görünür



def test_generated_code_also_goes_to_its_own_log(tmp_path):
    # Haiku'nun yazdığı kodlar ayrı dosyada: işe yarayanlar kalıcı tool'a çevrilmek için buradan taranır.
    client = make_client(tmp_path, [
        AIMessage("", tool_calls=[{"name": "request_new_tool", "id": "1", "args": {"gorev": "haftalık ders saati"}}]),
        AIMessage(ANSWER),
    ], prompt_version="agent_v7", code_runner=FakeCodeRunner())
    client.post("/agent/ask", json={"question": "3. sınıfın haftada kaç saat dersi var?"})
    agent_record = json.loads((tmp_path / "agent.jsonl").read_text(encoding="utf-8"))
    [record] = [json.loads(line) for line in (tmp_path / "anlik_tool.jsonl").read_text(encoding="utf-8").splitlines()]
    assert record["id"] == agent_record["id"]
    assert record["soru"] == "3. sınıfın haftada kaç saat dersi var?"
    assert record["kodlar"] == [{"kod": "print(21)", "cikti": "21\n"}]
    assert record["haiku_cevabi"] == "Toplam 21 saat."


def test_answer_without_code_writes_no_code_log(tmp_path):
    client = make_client(tmp_path, confirm_then_answer())
    client.post("/agent/ask", json={"question": QUESTION, "transcript": SAMPLE})
    assert not (tmp_path / "anlik_tool.jsonl").exists()

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


# --- Agent cevaplarında kaynak, geri bildirim ve yönetici kaydı (#16, #17) ---

from langchain_core.documents import Document  # noqa: E402

STAJ = Document(page_content="MADDE 10- (1) ... 40 iş günü ...",
                metadata={"chunk_id": 7, "source": "mühendislik fakültesi staj yönergesi.pdf",
                          "title": "mühendislik fakültesi staj yönergesi", "page": 3, "madde": 10})
STAJ_RETRIEVER = SimpleNamespace(chunks=[STAJ], search=lambda query: [STAJ])


def staj_client(tmp_path):
    return make_client(tmp_path, [
        AIMessage("", tool_calls=[{"name": "search_regulations", "id": "1", "args": {"query": "staj süresi"}}]),
        AIMessage("Zorunlu staj 40 iş günüdür [mühendislik fakültesi staj yönergesi, Madde 10]."),
    ], retriever=STAJ_RETRIEVER)


def test_agent_answer_carries_its_sources_and_logs_them(tmp_path):
    body = staj_client(tmp_path).post("/agent/ask", json={"question": "Zorunlu staj kaç iş günü?"}).json()
    assert body["sources"] == [{"chunk_id": 7, "label": "mühendislik fakültesi staj yönergesi, Madde 10, s. 3",
                                "source": "mühendislik fakültesi staj yönergesi.pdf",
                                "title": "mühendislik fakültesi staj yönergesi", "page": 3, "madde": 10}]
    assert json.loads(log_text(tmp_path))["kaynaklar"][0]["chunk_id"] == 7


def test_agent_answer_accepts_feedback(tmp_path):
    client = staj_client(tmp_path)
    answer_id = client.post("/agent/ask", json={"question": "Zorunlu staj kaç iş günü?"}).json()["id"]
    assert client.post("/feedback", json={"id": answer_id, "rating": 1}).status_code == 201


def test_admin_lists_agent_answers_with_agent_error_stages(tmp_path):
    client = staj_client(tmp_path)
    answer_id = client.post("/agent/ask", json={"question": "Zorunlu staj kaç iş günü?"}).json()["id"]
    service = app.dependency_overrides[get_service]()
    [row] = service.records()
    assert (row["id"], row["tur"], row["tool_lar"]) == (answer_id, "agent", ["search_regulations"])
    service.evaluate(answer_id, "yanlis", "evet", "tool_secimi")
    with pytest.raises(ValueError):
        service.evaluate(answer_id, "yanlis", "evet", "rewrite")      # agent'ta rewrite aşaması yok
    assert service.summary()["agent"] == 1
    service.delete(answer_id)
    assert service.records() == [] and log_text(tmp_path).count(answer_id) == 0


def test_plan_and_schedule_pdfs_are_served_only_as_agent_documents(tmp_path):
    client = make_client(tmp_path)
    service = app.dependency_overrides[get_service]()
    service.agent.documents = {"öğretim planı.pdf": "Öğretim Planı (data/öğretim planı.pdf)"}
    assert service.pdf_path("öğretim planı.pdf").name == "öğretim planı.pdf"
    with pytest.raises(KeyError):
        service.pdf_path("../.env")



def test_admin_skips_early_agent_records_without_question(tmp_path):
    # 2026-10-21'deki ilk agent kayıtları yalnız metadata tutuyordu (soru ve cevap yok); liste bunlarla bozulmasın.
    old = {"id": "eski", "zaman": "2026-10-21T18:00:00", "oturum": None, "transkript": True, "gecmis_uzunlugu": 0,
           "tool_lar": ["analyze_transcript"], "tur": 2, "config": "EXP-001", "commit": "abc", "latency_ms": 900,
           "hata": None}
    (tmp_path / "agent.jsonl").write_text(json.dumps(old) + "\n", encoding="utf-8")
    make_client(tmp_path)
    service = app.dependency_overrides[get_service]()
    assert service.records() == [] and service.summary()["toplam"] == 0
