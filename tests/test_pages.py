"""Sayfa rotaları ve ortak kabuk: HTML doğru dosyaları referans ediyor mu, gezinme etkin sayfayı işaretliyor mu.

Gerçek modeller yüklenmez (lifespan çalışmaz); sayfalar statik olduğu için service gerekmez.
"""
import json
import re

import pytest
from fastapi.testclient import TestClient

import config
from app.main import STATIC, app
from app.service import experiment_runs

# Sohbet, belgeler ve araçlar tek dokümandır: üç yol aynı kabuğu ve aynı üç görünümü döndürür, yalnız açılış görünümü değişir.
PAGES = [("/", "sohbet"), ("/belgeler", "belgeler"), ("/araclar", "araclar")]
ASSET = re.compile(r"""(?:href|src)="(/static/[^"?]+)\?v=[0-9a-f]{8}\"""")


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.mark.parametrize("path,active", PAGES)
def test_page_has_shell_and_marks_active_link(client, path, active):
    res = client.get(path)
    assert res.status_code == 200
    html = res.text
    assert "<!--KABUK" not in html and "<!--/KABUK" not in html and "{{BASLIK}}" not in html
    assert html.count('aria-current="page"') == 1
    assert f'data-nav="{active}" class="active" aria-current="page"' in html
    assert f'id="app" data-view="{active}"' in html
    for nav in ("sohbet", "belgeler", "araclar", "admin"):
        assert f'data-nav="{nav}"' in html
    for view in ("view-chat", "view-docs", "view-tools"):
        assert f'id="{view}"' in html
    # Değerlendirme yeni sekmede açılır (sohbet sekmesindeki transkript kaybolmasın).
    assert re.search(r'<a href="/admin"[^>]*target="_blank"[^>]*rel="noopener"', html)


def test_three_routes_share_the_same_body(client):
    def normalized(path):
        html = client.get(path).text
        html = re.sub(r' class="active" aria-current="page"|data-view="\w+"', "", html)
        return re.sub(r'(id="page-title">)[^<]*', r"", html)

    assert len({normalized(path) for path, _ in PAGES}) == 1


@pytest.mark.parametrize("path", [p for p, _ in PAGES])
def test_page_assets_are_versioned_and_exist(client, path):
    html = client.get(path).text
    assets = ASSET.findall(html)
    assert {"/static/shell.css", "/static/shell.js", "/static/chat.js", "/static/belgeler.js",
            "/static/araclar.js"} <= set(assets)
    assert "/static/logo-baibu.png" in assets
    for asset in assets:
        assert (STATIC / asset.removeprefix("/static/")).is_file(), asset
    assert client.get("/static/shell.css").status_code == 200


def test_admin_page_uses_shell_and_stays_protected(client, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "gizli")
    assert client.get("/admin").status_code == 401
    assert client.get("/admin/experiments").status_code == 401
    res = client.get("/admin", auth=("admin", "gizli"))
    assert res.status_code == 200
    assert 'data-nav="admin" class="active" aria-current="page"' in res.text
    assert "/static/admin.js" in res.text


def test_experiments_endpoint_returns_real_results(client, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "gizli")
    res = client.get("/admin/experiments", auth=("admin", "gizli"))
    assert res.status_code == 200
    assert [r["id"] for r in res.json()] == [r["id"] for r in experiment_runs()]


def test_experiment_runs_read_results_and_notes(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "configs").mkdir()
    result = {"id": "EXP-900", "calisma_zamani": "2026-01-02T10:00:00", "dataset": {"surum": "val_v1"},
              "agent": {"prompt_version": "agent_x"}, "environment": {"git_dirty": False},
              "llm": {"metrics": {"n": 4, "gerekli_tool_dogrulugu": 0.75, "tool_secimi_dogrulugu": 0.5,
                                  "agent_ms": 1234.5}}}
    (tmp_path / "results" / "EXP-900.json").write_text(json.dumps(result), encoding="utf-8")
    (tmp_path / "configs" / "EXP-900.json").write_text(json.dumps({"baslik": "b", "kisa_not": "not"}), encoding="utf-8")
    (tmp_path / "results" / "EXP-901.json").write_text(
        json.dumps({**result, "id": "EXP-901", "llm": None}), encoding="utf-8")  # agent metriği yok: atlanır
    assert experiment_runs(tmp_path) == [{
        "id": "EXP-900", "time": "2026-01-02T10:00:00", "dataset": "val_v1", "question_count": 4,
        "prompt": "agent_x", "title": "b", "note": "not", "needed_tool_acc": 0.75, "tool_choice_acc": 0.5,
        "mean_latency_ms": 1234.5, "git_dirty": False}]


def test_every_corpus_document_has_a_readable_title():
    titles = json.loads((STATIC / "belge_basliklari.json").read_text(encoding="utf-8"))
    corpus = [line.strip() for line in config.CORPORA["v1"].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert set(corpus) <= set(titles["belgeler"])
    assert all(info["grup"] in titles["gruplar"] for info in titles["belgeler"].values())


def test_tool_cards_cover_every_agent_tool():
    """arac_bilgisi.js agent'ın her tool'unu bir karta bağlar (kod adı kullanıcıya gösterilmez, eşleme eksik kalmasın)."""
    js = (STATIC / "arac_bilgisi.js").read_text(encoding="utf-8")
    mapped = set(re.findall(r'"([a-z_]+)"', " ".join(re.findall(r"names: \[([^\]]*)\]", js))))
    agent_src = (config.ROOT / "agent.py").read_text(encoding="utf-8")
    block = re.search(r"    tools = \((.*?)\)\n    return \[StructuredTool", agent_src, re.S).group(1)
    # tools = (...) bloğunda adı geçen ve agent.py'de `def` ile tanımlı olanlar tool'lardır (bayrak değişkenleri değil).
    agent_tools = {n for n in re.findall(r"[a-z]+(?:_[a-z]+)+", block) if re.search(rf"def {n}\(", agent_src)}
    assert len(agent_tools) >= 15
    assert agent_tools <= mapped, agent_tools - mapped
