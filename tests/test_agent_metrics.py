"""Deney hattının agent yolu: tool seçimi ve hesap doğruluğu metrikleri, koşucu ve rapor. Ağ ve LLM yok.

Agent sahte: soruya göre hazır cevap ve tool çağrıları döndürür. Deneme seti ve sahte transkript
tests/fixtures/agent_deneme/ altında (gerçek soru seti eval/'de, ayrı iş).
"""
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

import config
import graduation
from experiments import evaluation as ev
from experiments import report
from experiments import run

FIXTURES = Path(__file__).parent / "fixtures" / "agent_deneme"
TRANSCRIPTS = FIXTURES / "transkriptler"
TRIAL_SET = json.loads((FIXTURES / "sorular_deneme.json").read_text(encoding="utf-8"))
ITEMS = {q["id"]: q for q in TRIAL_SET["sorular"]}


# --- Argüman eşleşmesi ---

def test_expected_args_are_a_subset_of_actual_args():
    assert ev.args_match({"gun": "perşembe"}, {"gun": "perşembe", "sinif": 4})
    assert not ev.args_match({"gun": "perşembe", "sinif": 4}, {"gun": "perşembe"})


def test_string_args_ignore_case_and_whitespace():
    assert ev.args_match({"gun": "Perşembe"}, {"gun": "  perşembe "})
    assert ev.args_match({"ders": "İşletim  Sistemleri"}, {"ders": "işletim sistemleri"})
    assert not ev.args_match({"gun": "perşembe"}, {"gun": "cuma"})


def test_numeric_args_match_numeric_strings():
    assert ev.args_match({"sinif": 4}, {"sinif": "4"})
    assert ev.args_match({"sinif": 4}, {"sinif": 4.0})
    assert not ev.args_match({"sinif": 4}, {"sinif": 3})


def test_list_args_ignore_order_and_case():
    assert ev.args_match({"dersler": ["Yapay Zeka", "Görüntü İşleme"]},
                         {"dersler": ["görüntü işleme", "yapay zeka"]})
    assert not ev.args_match({"dersler": ["Yapay Zeka"]}, {"dersler": ["Yapay Zeka", "Görüntü İşleme"]})


# --- Tool seçimi ---

def calls(*names_args):
    return [{"name": n, "args": a} for n, a in names_args]


def test_right_tool_with_right_args_is_correct():
    out = ev.tool_selection([{"name": "get_course_schedule", "args": {"gun": "perşembe", "sinif": 4}}],
                            calls(("get_course_schedule", {"gun": "Perşembe", "sinif": 4})))
    assert out["tool_dogru"] == 1 and out["tool_adi_dogru"] == 1
    assert out["eksik_tool_lar"] == [] and out["fazla_tool_lar"] == []


def test_right_tool_with_wrong_args_counts_only_for_the_name():
    out = ev.tool_selection([{"name": "get_course_schedule", "args": {"gun": "perşembe"}}],
                            calls(("get_course_schedule", {"gun": "cuma"})))
    assert out["tool_dogru"] == 0 and out["tool_adi_dogru"] == 1
    assert out["eksik_tool_lar"] == ["get_course_schedule"]


def test_wrong_tool_is_incorrect():
    out = ev.tool_selection([{"name": "analyze_transcript"}], calls(("search_regulations", {"query": "mezuniyet"})))
    assert out["tool_dogru"] == 0 and out["tool_adi_dogru"] == 0
    assert out["eksik_tool_lar"] == ["analyze_transcript"] and out["fazla_tool_lar"] == ["search_regulations"]


def test_unexpected_extra_tool_makes_selection_incorrect():
    out = ev.tool_selection([{"name": "analyze_transcript"}],
                            calls(("analyze_transcript", {}), ("search_regulations", {"query": "GNO"})))
    assert out["tool_dogru"] == 0 and out["tool_adi_dogru"] == 0
    assert out["fazla_tool_lar"] == ["search_regulations"]


def test_repeating_an_expected_tool_is_allowed():
    out = ev.tool_selection([{"name": "search_regulations"}],
                            calls(("search_regulations", {"query": "staj"}), ("search_regulations", {"query": "staj süresi"})))
    assert out["tool_dogru"] == 1


def test_expecting_no_tool():
    assert ev.tool_selection([], [])["tool_dogru"] == 1
    assert ev.tool_selection([], calls(("search_regulations", {"query": "merhaba"})))["tool_dogru"] == 0


def test_several_expected_tools_must_all_be_called_in_any_order():
    expected = [{"name": "confirm_equivalence"}, {"name": "analyze_transcript"}]
    assert ev.tool_selection(expected, calls(("analyze_transcript", {}), ("confirm_equivalence", {})))["tool_dogru"] == 1
    assert ev.tool_selection(expected, calls(("analyze_transcript", {})))["tool_dogru"] == 0


# --- Hesap doğruluğu ---

def test_required_tools_ignore_harmless_extra_calls():
    # EXP-003: kuralın metni için ek search_regulations doğru bir davranıştı ama katı metrik onu hata saydı.
    out = ev.tool_selection([{"name": "analyze_transcript"}],
                            calls(("search_regulations", {"query": "staj"}), ("analyze_transcript", {})))
    assert (out["tool_dogru"], out["gerekli_tool_dogru"], out["yan_etkisiz"]) == (0, 1, 1)


def test_required_tools_still_need_the_expected_args():
    out = ev.tool_selection([{"name": "get_course_schedule", "args": {"gun": "perşembe"}}],
                            calls(("get_course_schedule", {"gun": "cuma"})))
    assert out["gerekli_tool_dogru"] == 0


def test_unexpected_side_effect_tool_is_flagged():
    # Otomatik eşlenmiş Erasmus dersi sorulunca istenmemiş teyit kaydı: oturum durumunu değiştirir.
    out = ev.tool_selection([{"name": "analyze_transcript"}],
                            calls(("analyze_transcript", {}), ("confirm_equivalence", {"transcript_course": "X"})))
    assert (out["gerekli_tool_dogru"], out["yan_etkisiz"]) == (1, 0)


def test_expected_side_effect_tool_is_not_flagged():
    out = ev.tool_selection([{"name": "confirm_equivalence"}], calls(("confirm_equivalence", {})))
    assert out["yan_etkisiz"] == 1


def test_new_tool_metrics_are_derived_from_older_result_rows():
    # Eski sonuç dosyaları değişmez; yeni metrikler kayıtlı eksik/fazla tool listelerinden türetilir.
    row = {"tool_dogru": 0, "eksik_tool_lar": [], "fazla_tool_lar": ["confirm_equivalence"]}
    assert run.per_question_value(row, "gerekli_tool_dogrulugu", None) == 1
    assert run.per_question_value(row, "yan_etkisiz_orani", None) == 0
    assert run.per_question_value({"answer": "x"}, "gerekli_tool_dogrulugu", None) is None
    metrics = run.derived_agent_metrics([row, {"eksik_tool_lar": ["analyze_transcript"], "fazla_tool_lar": []}])
    assert (metrics["gerekli_tool_dogrulugu"], metrics["yan_etkisiz_orani"]) == (0.5, 0.5)


def test_numbers_in_text_reads_turkish_and_english_decimals():
    assert ev.numbers_in_text("Geçilen 180 AKTS, GNO 2,47; bu dönem sonunda 2.47 kalır.") == [180.0, 2.47, 2.47]


def test_calculation_is_correct_when_every_expected_number_is_in_the_answer():
    out = ev.calculation({"gecilen_akts": 12, "gno": 2.75}, "Şu an 12 AKTS geçilmiş, GNO 2,75.")
    assert out["hesap_dogru"] == 1
    assert out["sayilar"] == {"gecilen_akts": {"beklenen": 12, "cevapta": True},
                              "gno": {"beklenen": 2.75, "cevapta": True}}


def test_calculation_requires_exact_match():
    out = ev.calculation({"gecilen_akts": 12, "gno": 2.75}, "Şu an 12 AKTS geçilmiş, GNO yaklaşık 2,8.")
    assert out["hesap_dogru"] == 0
    assert out["sayilar"]["gno"]["cevapta"] is False


def test_trailing_zero_in_gno_still_matches():
    assert ev.calculation({"gno": 2.5}, "GNO 2,50")["hesap_dogru"] == 1


def test_reference_calculation_checks_the_deterministic_module_separately():
    reference = {"gecilen_akts": 12, "gno": 2.75, "devam_edenler_gecilirse_akts": 15}
    out = ev.calculation({"gecilen_akts": 12, "devam_edenler_gecilirse_akts": 14}, "12 ve 14", reference)
    assert out["hesap_dogru"] == 1
    assert out["arac_hesap_dogru"] == 0
    assert out["sayilar"]["devam_edenler_gecilirse_akts"]["aracta"] == 15


def test_reference_without_the_expected_keys_is_not_judged():
    out = ev.calculation({"simulasyon_akts": 30}, "30 AKTS", {"gecilen_akts": 12})
    assert out["arac_hesap_dogru"] is None


# --- Soru seti doğrulaması ---

def test_trial_set_is_valid():
    assert ev.validate_agent_items(TRIAL_SET, TRANSCRIPTS) == []


def test_agent_item_problems_are_reported():
    data = {"sorular": [
        {"id": "X1", "soru": "?", "beklenen_tool_lar": [{"args": {}}]},
        {"id": "X2", "soru": "?", "transkript": "yok"},
        {"id": "X3", "soru": "?", "beklenen_sayilar": {"gecilen_akts": 1}},
        {"id": "X4", "soru": "?", "transkript": "ornek_2023", "beklenen_sayilar": {"gno": "iki"}},
    ]}
    problems = ev.validate_agent_items(data, TRANSCRIPTS)
    assert len(problems) == 4
    assert [p.split(":")[0] for p in problems] == ["X1", "X2", "X3", "X4"]


def test_transcript_fixture_with_personal_fields_is_rejected(tmp_path):
    fixture = json.loads((TRANSCRIPTS / "ornek_2023.json").read_text(encoding="utf-8"))
    (tmp_path / "kisisel.json").write_text(json.dumps({**fixture, "ad_soyad": "X Y"}), encoding="utf-8")
    problems = ev.validate_agent_items({"sorular": [{"id": "X1", "soru": "?", "transkript": "kisisel"}]}, tmp_path)
    assert problems and "ad_soyad" in problems[0]


# --- Koşucu: agent yolu ---

class FakeAgent:
    """Soruya göre hazır sonuç döndürür ve aldığı argümanları kaydeder (AdvisorAgent.ask ile aynı imza)."""

    def __init__(self, replies):
        self.replies = replies
        self.received = []

    def ask(self, question, history=None, transcript=None, on_tool_call=None, confirmations=None):
        self.received.append({"question": question, "history": history, "transcript": transcript,
                              "confirmations": confirmations})
        reply = self.replies[question]
        if isinstance(reply, Exception):
            raise reply
        answer, tool_calls = reply
        return {"answer": answer, "tool_calls": tool_calls, "turns": len(tool_calls) + 1,
                "usage": {"input": 100, "output": 20, "reasoning": 0}}


REPLIES = {
    ITEMS["D01"]["soru"]: ("Staj başvurusu bölüm staj komisyonuna yapılır [Staj Yönergesi, Madde 5].",
                           calls(("search_regulations", {"query": "staj başvurusu"}))),
    ITEMS["D02"]["soru"]: ("Şu an 12 AKTS geçilmiş; bu dönemki ders geçilirse 14 AKTS olur. GNO 2,75.",
                           calls(("analyze_transcript", {}))),
    ITEMS["D03"]["soru"]: ("Perşembe 4. sınıfın dersleri: ...",
                           calls(("get_course_schedule", {"gun": "Cuma", "sinif": 4}))),  # yanlış gün
    ITEMS["D04"]["soru"]: ("Bu dönemki ders geçilirse 16 AKTS olur.",  # yanlış sayı
                           calls(("analyze_transcript", {}))),
}


def run_trial(replies=REPLIES):
    agent = FakeAgent(replies)
    rows, complete = run.agent_rows(agent, TRIAL_SET["sorular"], wait=0, plan=graduation.load_plan(),
                                    transcripts_dir=TRANSCRIPTS)
    return agent, {r["id"]: r for r in rows}, complete


def test_agent_rows_record_tool_calls_and_arguments():
    _, rows, complete = run_trial()
    assert complete
    assert rows["D01"]["tool_calls"] == [{"name": "search_regulations", "args": {"query": "staj başvurusu"}}]
    assert rows["D03"]["tool_calls"] == [{"name": "get_course_schedule", "args": {"gun": "Cuma", "sinif": 4}}]
    assert rows["D02"]["turns"] == 2 and rows["D02"]["tokens"]["input"] == 100


def test_agent_rows_pass_transcript_history_and_fresh_confirmations():
    agent, _, _ = run_trial()
    by_question = {r["question"]: r for r in agent.received}
    fixture = json.loads((TRANSCRIPTS / "ornek_2023.json").read_text(encoding="utf-8"))
    assert by_question[ITEMS["D01"]["soru"]]["transcript"] is None
    assert by_question[ITEMS["D02"]["soru"]]["transcript"] == fixture
    assert by_question[ITEMS["D04"]["soru"]]["history"] == ITEMS["D04"]["gecmis"]
    assert by_question[ITEMS["D02"]["soru"]]["confirmations"] == {}


def test_agent_rows_score_each_question():
    _, rows, _ = run_trial()
    assert rows["D01"]["tool_dogru"] == 1 and rows["D01"]["answer_ok"] == 1
    assert rows["D02"]["tool_dogru"] == 1 and rows["D02"]["hesap_dogru"] == 1
    assert rows["D02"]["arac_hesap_dogru"] == 1  # elle hesaplanan sayılar graduation.evaluate ile aynı
    assert rows["D03"]["tool_dogru"] == 0 and rows["D03"]["tool_adi_dogru"] == 1
    assert rows["D04"]["hesap_dogru"] == 0
    # Değerlendirme ölçütü olmayan alanlar yazılmaz (ortalamaya girmesin).
    assert "hesap_dogru" not in rows["D01"] and "answer_ok" not in rows["D02"]


def test_agent_metrics_average_only_questions_that_define_them():
    _, rows, _ = run_trial()
    m = run.agent_metrics(list(rows.values()))
    assert m["tool_secimi_dogrulugu"] == 0.75 and m["tool_secimi_n"] == 4
    assert m["tool_adi_dogrulugu"] == 1.0
    assert m["hesap_dogrulugu"] == 0.5 and m["hesap_n"] == 2
    assert m["arac_hesap_dogrulugu"] == 1.0
    assert m["answer_acc"] == 1.0 and m["answer_n"] == 1
    assert m["ort_tool_cagrisi"] == 1.0 and m["ort_tur"] == 2.0


def test_quota_error_stops_the_agent_run():
    replies = {**REPLIES, ITEMS["D02"]["soru"]: RuntimeError("429 ResourceExhausted")}
    _, rows, complete = run_trial(replies)
    assert not complete and list(rows) == ["D01"]


def test_other_agent_errors_are_recorded_and_the_run_continues():
    replies = {**REPLIES, ITEMS["D02"]["soru"]: TimeoutError("zaman aşımı")}
    _, rows, complete = run_trial(replies)
    assert complete and rows["D02"]["answer"] == "<hata: TimeoutError>"
    assert rows["D02"]["tool_calls"] == [] and rows["D02"]["hesap_dogru"] == 0


def test_real_agent_receives_the_fixture_transcript():
    """Koşucunun AdvisorAgent.ask ile uyumu: analyze_transcript fixture'ı gerçek modülle değerlendiriyor."""
    from agent import AdvisorAgent

    class Scripted:
        def __init__(self):
            self.replies = [AIMessage("", tool_calls=[{"name": "analyze_transcript", "args": {}, "id": "1"}]),
                            AIMessage("12 AKTS geçilmiş, 14 AKTS olur, GNO 2,75.")]
            self.received = []

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            self.received.append(list(messages))
            return self.replies.pop(0)

    llm = Scripted()
    plan = graduation.load_plan()
    rows, _ = run.agent_rows(AdvisorAgent(retriever=None, llm=llm, plan=plan), [ITEMS["D02"]], wait=0, plan=plan,
                             transcripts_dir=TRANSCRIPTS)
    tool_message = next(m for m in llm.received[-1] if isinstance(m, ToolMessage))
    assert "12" in tool_message.content
    assert rows[0]["tool_dogru"] == 1 and rows[0]["hesap_dogru"] == 1


# --- Uçtan uca: config → koşu → sonuç dosyası → rapor ---

AGENT_EXP = {
    "id": "EXP-900", "baslik": "Deneme agent koşusu", "ust_deney": None, "veri_seti": "sorular_deneme",
    "neden": "test", "hipotez": "test", "birincil_metrik": "tool_secimi_dogrulugu", "beklenen_yon": "artis",
    "params": {**json.loads((config.EXPERIMENTS_DIR / "configs" / "EXP-001.json").read_text(encoding="utf-8"))["params"],
               "mod": "agent", "prompt_version": "agent_v1", "llm_model": "gemini-3.5-flash-lite"},
    "kisa_not": "deneme",
}


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Config, sonuç ve günlük geçici klasörde; soru seti deneme seti; agent sahte; ortam bilgisi sabit."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "EXP-900.json").write_text(json.dumps(AGENT_EXP, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(config, "EXPERIMENTS_DIR", tmp_path)
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(report, "LOG_FILE", tmp_path / "experiment_log.md")
    monkeypatch.setattr(config, "load_experiment",
                        lambda exp_id: json.loads((tmp_path / "configs" / f"{exp_id}.json").read_text(encoding="utf-8")))
    monkeypatch.setattr(ev, "dataset_path", lambda name=None: FIXTURES / f"{name}.json")
    monkeypatch.setattr(ev, "TRANSCRIPTS_DIR", TRANSCRIPTS)
    monkeypatch.setattr(ev, "environment", lambda cfg: {"git_commit": "test", "git_dirty": False})
    agent = FakeAgent(REPLIES)
    monkeypatch.setattr(run, "make_agent", lambda exp_id, cfg: (agent, {"chunks": 123}))
    datasets = lambda name: ev.load_dataset(name)
    return tmp_path, datasets


def test_agent_run_writes_result_file_with_tool_calls(sandbox):
    tmp_path, datasets = sandbox
    run.run("EXP-900", use_llm=True, wait=0, rerun=False, datasets=datasets)
    result = json.loads((tmp_path / "results" / "EXP-900.json").read_text(encoding="utf-8"))
    assert result["retrieval"] is None
    assert result["params"]["mod"] == "agent"
    assert result["dataset"]["transkriptler"] == {"ornek_2023": ev.file_hash(TRANSCRIPTS / "ornek_2023.json")}
    assert result["agent"]["plan_hash"] == ev.file_hash(config.PLAN_PATH)
    llm = result["llm"]
    assert llm["tamamlandi"] and llm["metrics"]["tool_secimi_dogrulugu"] == 0.75
    assert [r["tool_calls"][0]["name"] for r in llm["per_question"]] == [
        "search_regulations", "analyze_transcript", "get_course_schedule", "analyze_transcript"]
    assert llm["alt_kirilimlar"]["grup:transkript"]["hesap_dogrulugu"] == 0.5


def test_agent_run_without_llm_flag_is_skipped(sandbox, capsys):
    tmp_path, datasets = sandbox
    run.run("EXP-900", use_llm=False, wait=0, rerun=False, datasets=datasets)
    assert "--llm" in capsys.readouterr().out
    assert not (tmp_path / "results" / "EXP-900.json").exists()


def test_report_shows_agent_metrics(sandbox):
    tmp_path, datasets = sandbox
    run.run("EXP-900", use_llm=True, wait=0, rerun=False, datasets=datasets)
    report.build()
    log = (tmp_path / "experiment_log.md").read_text(encoding="utf-8")
    assert "Agent" in log
    row = next(l for l in log.splitlines() if l.startswith("| EXP-900 |") and "0.750" in l)
    assert "0.500" in row and "1.000" in row


def test_agent_follow_up_compares_new_metrics_with_parent(sandbox, monkeypatch):
    tmp_path, datasets = sandbox
    run.run("EXP-900", use_llm=True, wait=0, rerun=False, datasets=datasets)
    child = {**AGENT_EXP, "id": "EXP-901", "ust_deney": "EXP-900",
             "params": {**AGENT_EXP["params"], "llm_model": "gemini-3.1-flash-lite"}}
    (tmp_path / "configs" / "EXP-901.json").write_text(json.dumps(child, ensure_ascii=False), encoding="utf-8")
    better = {**REPLIES, ITEMS["D03"]["soru"]: ("...", calls(("get_course_schedule", {"gun": "perşembe", "sinif": 4})))}
    monkeypatch.setattr(run, "make_agent", lambda exp_id, cfg: (FakeAgent(better), {"chunks": 123}))
    run.run("EXP-901", use_llm=True, wait=0, rerun=False, datasets=datasets)
    comp = json.loads((tmp_path / "results" / "EXP-901.json").read_text(encoding="utf-8"))["karsilastirma"]
    assert comp["ayni_veri_seti"]
    m = comp["metrikler"]["tool_secimi_dogrulugu"]
    assert m["once"] == 0.75 and m["sonra"] == 1.0 and m["test"] == "McNemar (kesin)"
    assert comp["karar"]["metrik"] == "tool_secimi_dogrulugu"
    assert "degisen_sorular" not in comp


# --- RAG yolu değişmedi ---

def test_old_configs_default_to_rag_mode():
    exp = json.loads((config.EXPERIMENTS_DIR / "configs" / "EXP-001.json").read_text(encoding="utf-8"))
    assert "mod" not in exp["params"]
    assert config.RagConfig.from_dict(exp["params"]).mod == "rag"


def test_unknown_mode_is_rejected():
    exp = json.loads((config.EXPERIMENTS_DIR / "configs" / "EXP-001.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="mod"):
        config.RagConfig.from_dict({**exp["params"], "mod": "otomatik"})


def test_rag_comparison_is_unchanged():
    """RAG sonuçlarında karşılaştırma eskisi gibi: retrieval + llm metrikleri ve sırası değişen sorular."""
    def result(ranks, oks):
        rows = [{"id": f"Q{i}", "rank": r, "rr": 1 / r if r else 0.0, "hit@1": int(r == 1), "hit@3": int(bool(r and r <= 3)),
                 "hit@5": int(bool(r and r <= 5)), "hit@10": int(bool(r and r <= 10)), "context_hit": int(bool(r and r <= 5)),
                 "context_chars": 1000, "latency_ms": 10.0, "asamalar": {"final": r}} for i, r in enumerate(ranks)]
        answers = [{"id": f"Q{i}", "answer": "x", "answer_ok": ok} for i, ok in enumerate(oks)]
        return {"dataset": {"hash": "h"}, "retrieval": {"per_question": rows}, "llm": {"per_question": answers}}

    exp = {"ust_deney": "EXP-A", "birincil_metrik": "mrr@10", "beklenen_yon": "artis"}
    comp = run.compare(exp, result([1, 2, None], [1, 0, 1]), result([2, 2, None], [1, 1, 1]))
    assert comp["metrikler"]["mrr@10"]["fark"] == pytest.approx(0.5 / 3, abs=1e-4)
    assert comp["metrikler"]["answer_acc"]["fark"] == pytest.approx(-1 / 3, abs=1e-4)
    assert comp["degisen_sorular"] == [{"id": "Q0", "once": 2, "sonra": 1}]
    assert "tool_secimi_dogrulugu" not in comp["metrikler"]
