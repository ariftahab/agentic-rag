"""Anlık tool (code as action): sahte bir Anthropic client'ıyla, ağ ve LLM yok.

Korunan şeyler: dosyalar sandbox'a container_upload ile gider ve istekten sonra (hata olsa da) silinir, her soru
yeni bir container'da başlar, pause_turn'de aynı container'la devam edilir, üretilen kod ve çıktısı toplanır.
"""
from types import SimpleNamespace as NS

import pytest

from tools import codeact


def text(t):
    return NS(type="text", text=t)


def code_call(command):
    return NS(type="server_tool_use", name="bash_code_execution", input={"command": command})


def code_result(stdout, return_code=0, stderr=""):
    return NS(type="bash_code_execution_tool_result",
              content=NS(type="bash_code_execution_result", stdout=stdout, stderr=stderr, return_code=return_code))


def response(*blocks, stop_reason="end_turn", container_id="cnt_1"):
    return NS(content=list(blocks), stop_reason=stop_reason, container=NS(id=container_id))


class FakeClient:
    def __init__(self, responses, fail=False):
        self.responses = list(responses)
        self.fail = fail
        self.uploaded, self.deleted, self.requests = [], [], []
        self.files = NS(upload=self._upload, delete=self.deleted.append)
        self.messages = NS(create=self._create)

    def _upload(self, file):
        name, data, mime = file
        self.uploaded.append((name, data, mime))
        return NS(id=f"file_{len(self.uploaded)}")

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if self.fail:
            raise RuntimeError("API hatası")
        return self.responses.pop(0)


def test_files_go_to_the_sandbox_and_are_deleted_afterwards():
    client = FakeClient([response(code_call("python -c 'print(12)'"), code_result("12\n"), text("Toplam 12 saat."))])
    result = codeact.CodeRunner(client).run("3. sınıfın haftalık ders saati", {"ders_programi.json": "{}"})
    request = client.requests[0]
    assert request["model"] == codeact.CODE_MODEL and request["tools"] == [codeact.CODE_TOOL]
    assert "container" not in request                                  # her soru yeni container
    assert {"type": "container_upload", "file_id": "file_1"} in request["messages"][0]["content"]
    assert client.uploaded == [("ders_programi.json", b"{}", "application/json")]
    assert client.deleted == ["file_1"]
    assert result["answer"] == "Toplam 12 saat."
    assert result["steps"] == [{"tip": "kod", "kod": "python -c 'print(12)'", "cikti": "12\n"},
                               {"tip": "metin", "metin": "Toplam 12 saat."}]
    assert (result["requests"], result["stop_reason"]) == (1, "end_turn") and result["duration_ms"] >= 0


def test_files_are_deleted_even_when_the_request_fails():
    client = FakeClient([], fail=True)
    with pytest.raises(RuntimeError):
        codeact.CodeRunner(client).run("görev", {"a.json": "{}", "b.json": "{}"})
    assert client.deleted == ["file_1", "file_2"]


def test_pause_turn_continues_in_the_same_container_within_a_limit():
    paused = response(code_call("ls"), code_result("a.json\n"), stop_reason="pause_turn", container_id="cnt_9")
    client = FakeClient([paused, response(text("Bitti."))])
    result = codeact.CodeRunner(client).run("görev", {})
    assert client.requests[1]["container"] == "cnt_9"
    assert client.requests[1]["messages"][-1] == {"role": "assistant", "content": paused.content}
    assert result["answer"] == "Bitti." and result["steps"][0]["kod"] == "ls"


def test_endless_pause_turn_stops_after_the_limit():
    client = FakeClient([response(stop_reason="pause_turn")] * (codeact.MAX_REQUESTS + 2))
    result = codeact.CodeRunner(client).run("görev", {})
    assert len(client.requests) == codeact.MAX_REQUESTS
    assert result["answer"] == codeact.UNFINISHED


def test_failed_code_reports_stderr():
    client = FakeClient([response(code_call("python x.py"), code_result("", 1, "NameError"), text("Hata aldım."))])
    assert codeact.CodeRunner(client).run("görev", {})["steps"][0]["cikti"] == "Hata (çıkış kodu 1): NameError"


def test_steps_keep_the_order_of_moves_and_sum_tokens_across_requests():
    # Yanlış bir cevapta hangi hamlenin bozduğunu görmek için: kod → çıktı → ara metin → kod → çıktı, sırasıyla.
    first = response(code_call("ls"), code_result("a.json\n"), text("Dosyaya bakıyorum."),
                     stop_reason="pause_turn")
    first.usage = NS(input_tokens=100, output_tokens=10)
    second = response(code_call("python hesap.py"), code_result("21\n"), text("21 saat."))
    second.usage = NS(input_tokens=50, output_tokens=5)
    result = codeact.CodeRunner(FakeClient([first, second])).run("görev", {})
    assert [s["tip"] for s in result["steps"]] == ["kod", "metin", "kod", "metin"]
    assert [s.get("cikti") for s in result["steps"]] == ["a.json\n", None, "21\n", None]
    assert result["tokens"] == {"input": 150, "output": 15} and result["requests"] == 2


def test_task_message_describes_each_file_so_haiku_skips_exploring():
    client = FakeClient([response(text("ok"))])
    data = '{"donem": "2026-2027 Güz", "dersler": [{"gun": "Pazartesi", "sinif": 1}, {"gun": "Salı", "sinif": 2}]}'
    codeact.CodeRunner(client).run("görev", {"ders_programi.json": data})
    message = client.requests[0]["messages"][0]["content"][0]["text"]
    assert "$INPUT_DIR/ders_programi.json" in message
    assert '"donem": "2026-2027 Güz"' in message                       # kısa değerler olduğu gibi
    assert "dersler: 2 kayıtlı liste" in message and '{"gun": "Pazartesi", "sinif": 1}' in message  # ilk kayıt örnek


# --- Devir (agent_v9): döngü aynı mesaj listesiyle Haiku'yla devam eder ---

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402


def tool_use(name, args, id_="tu_1"):
    return NS(type="tool_use", id=id_, name=name, input=args)


class SearchTool:
    name = "search_regulations"
    description = "Mevzuatta arama."
    args_schema = NS(model_json_schema=lambda: {"type": "object", "properties": {"query": {"type": "string"}}})


CONVERSATION = [
    SystemMessage("system"), HumanMessage("62 aldım, ortalamam?"),
    AIMessage("", tool_calls=[{"name": "search_regulations", "args": {"query": "not tablosu"}, "id": "a/b 1"}]),
    ToolMessage("Madde 27 tablosu", tool_call_id="a/b 1"),
    AIMessage("", tool_calls=[{"name": "request_new_tool", "args": {"gorev": "ortalama"}, "id": "c2"}]),
    ToolMessage("Soru devredildi.", tool_call_id="c2"),
]


def test_handoff_continues_the_same_conversation_with_agent_tools():
    client = FakeClient([response(tool_use("search_regulations", {"query": "katsayılar"}), stop_reason="tool_use"),
                         response(code_call("python3 hesap.py"), code_result("3.06\n"), text("GNO 3,06."))])
    executed = []
    result = codeact.CodeRunner(client).handoff(
        "kurallar", CONVERSATION, [SearchTool()], lambda name, args: executed.append((name, args)) or "tablo",
        {"transkript.json": '{"dersler": []}'})
    first = client.requests[0]
    # Prompt caching: system sonunda açık breakpoint, büyüyen konuşma için otomatik breakpoint.
    assert first["system"] == [{"type": "text", "text": "kurallar", "cache_control": {"type": "ephemeral"}}]
    assert first["cache_control"] == {"type": "ephemeral"}
    roles = [m["role"] for m in first["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]   # system ayrı, roller sırayla
    uses = [b for m in first["messages"] for b in m["content"] if b["type"] == "tool_use"]
    assert [u["id"] for u in uses] == ["a_b_1", "c2"]                    # Anthropic'in kabul ettiği id biçimi
    last = first["messages"][-1]["content"]
    assert [b["type"] for b in last] == ["tool_result", "text", "container_upload"]
    assert {t["name"] for t in first["tools"]} == {"search_regulations", "code_execution"}
    assert executed == [("search_regulations", {"query": "katsayılar"})]
    assert client.requests[1]["container"] == "cnt_1"
    assert client.requests[1]["messages"][-1]["content"][0]["content"] == "tablo"
    assert result["answer"] == "GNO 3,06." and result["tool_calls"] == [{"name": "search_regulations",
                                                                         "args": {"query": "katsayılar"}}]
    assert [s["tip"] for s in result["steps"]] == ["tool", "kod", "metin"] and result["steps"][1]["cikti"] == "3.06\n"
    assert client.deleted == ["file_1"]


def test_handoff_stops_after_the_request_limit():
    client = FakeClient([response(tool_use("search_regulations", {"query": "x"}), stop_reason="tool_use")
                         for _ in range(codeact.MAX_HANDOFF_REQUESTS)])
    result = codeact.CodeRunner(client).handoff("s", CONVERSATION, [SearchTool()], lambda n, a: "r", {})
    assert result["answer"] == codeact.UNFINISHED and len(client.requests) == codeact.MAX_HANDOFF_REQUESTS


# --- Denetim (agent_h7): taslak cevap ayrı bir çağrıyla denetlenir, "eksik"te aynı konuşma devam eder ---

import json  # noqa: E402


def verdict(karar, eksikler=(), tools=()):
    return response(text(json.dumps({"karar": karar, "eksikler": list(eksikler), "onerilen_tool_lar": list(tools),
                                     "gerekce": "g"}, ensure_ascii=False)))


QUESTION = [SystemMessage("system"), HumanMessage("Ortak zorunlu derslerim hangileri?")]


def test_review_sends_feedback_and_the_agent_continues_until_approved():
    client = FakeClient([response(text("8 zorunlu dersin var.")),
                         verdict("eksik", ["ortak zorunlu tanımı aranmadı"], ["search_regulations"]),
                         response(tool_use("search_regulations", {"query": "ortak zorunlu"}), stop_reason="tool_use"),
                         response(text("Ortak zorunlu 3 ders: AİİT, Türk Dili, Yabancı Dil.")),
                         verdict("tamam")])
    result = codeact.CodeRunner(client).handoff("s", QUESTION, [SearchTool()], lambda n, a: "Madde 20/4", {},
                                                review_rounds=3, question="Ortak zorunlu derslerim hangileri?")
    review_request = client.requests[1]
    assert review_request["output_config"]["format"]["schema"] == codeact.REVIEW_SCHEMA
    assert "tools" not in review_request                                 # denetleyici tool çağırmaz
    assert "8 zorunlu dersin var." in review_request["messages"][0]["content"]
    note = client.requests[2]["messages"][-1]
    assert note["role"] == "user" and "ortak zorunlu tanımı aranmadı" in note["content"][0]["text"]
    second_review = client.requests[4]["messages"][0]["content"]
    assert "search_regulations" in second_review and "Madde 20/4" in second_review  # tool sonucu denetlenir
    assert result["answer"] == "Ortak zorunlu 3 ders: AİİT, Türk Dili, Yabancı Dil."
    assert [(r["tur"], r["karar"], r["sonra_cagrilan"]) for r in result["review"]] == [
        (1, "eksik", ["search_regulations"]), (2, "tamam", [])]
    assert result["review"][0]["taslak"] == "8 zorunlu dersin var."


def test_review_stops_after_the_round_limit_and_keeps_the_last_revision():
    replies = []
    for i in range(codeact_rounds := 3):
        replies += [response(text(f"taslak {i}")), verdict("eksik", ["x"])]
    replies.append(response(text("son düzeltme")))
    client = FakeClient(replies)
    result = codeact.CodeRunner(client).handoff("s", QUESTION, [SearchTool()], lambda n, a: "r", {},
                                                review_rounds=codeact_rounds, question="q")
    assert len(result["review"]) == codeact_rounds and result["answer"] == "son düzeltme"
    assert len(client.requests) == 2 * codeact_rounds + 1                  # son düzeltme denetlenmez


def test_unparseable_review_keeps_the_draft():
    client = FakeClient([response(text("taslak")), response(text("bozuk"))])
    result = codeact.CodeRunner(client).handoff("s", QUESTION, [SearchTool()], lambda n, a: "r", {},
                                                review_rounds=3, question="q")
    assert result["answer"] == "taslak" and result["review"][0]["karar"] == "tamam"


def test_without_review_rounds_no_review_call_is_made():
    client = FakeClient([response(text("cevap"))])
    result = codeact.CodeRunner(client).handoff("s", QUESTION, [SearchTool()], lambda n, a: "r", {})
    assert len(client.requests) == 1 and result["review"] == []
