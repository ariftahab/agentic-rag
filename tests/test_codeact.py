"""Anlık tool (code as action): sahte bir Anthropic client'ıyla, ağ ve LLM yok.

Korunan şeyler: dosyalar sandbox'a container_upload ile gider ve istekten sonra (hata olsa da) silinir, her soru
yeni bir container'da başlar, pause_turn'de aynı container'la devam edilir, üretilen kod ve çıktısı toplanır.
"""
from types import SimpleNamespace as NS

import pytest

import codeact


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
    assert result == {"answer": "Toplam 12 saat.", "code": ["python -c 'print(12)'"], "outputs": ["12\n"]}


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
    assert result["answer"] == "Bitti." and result["code"] == ["ls"]


def test_endless_pause_turn_stops_after_the_limit():
    client = FakeClient([response(stop_reason="pause_turn")] * (codeact.MAX_REQUESTS + 2))
    result = codeact.CodeRunner(client).run("görev", {})
    assert len(client.requests) == codeact.MAX_REQUESTS
    assert result["answer"] == codeact.UNFINISHED


def test_failed_code_reports_stderr():
    client = FakeClient([response(code_call("python x.py"), code_result("", 1, "NameError"), text("Hata aldım."))])
    assert codeact.CodeRunner(client).run("görev", {})["outputs"] == ["Hata (çıkış kodu 1): NameError"]


def test_task_message_describes_each_file_so_haiku_skips_exploring():
    client = FakeClient([response(text("ok"))])
    data = '{"donem": "2026-2027 Güz", "dersler": [{"gun": "Pazartesi", "sinif": 1}, {"gun": "Salı", "sinif": 2}]}'
    codeact.CodeRunner(client).run("görev", {"ders_programi.json": data})
    message = client.requests[0]["messages"][0]["content"][0]["text"]
    assert "$INPUT_DIR/ders_programi.json" in message
    assert '"donem": "2026-2027 Güz"' in message                       # kısa değerler olduğu gibi
    assert "dersler: 2 kayıtlı liste" in message and '{"gun": "Pazartesi", "sinif": 1}' in message  # ilk kayıt örnek
