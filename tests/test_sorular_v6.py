"""Soru seti v6 (eval/sorular_v6.json): v5 ile aynı sorular ve cevaplar; yalnız kalıcı tool'a taşınan 8 sorunun
beklenen tool'ları değişti (agent_h6). Ağ ve LLM yok."""
from experiments import evaluation as ev

V6, _ = ev.load_dataset("sorular_v6")
V5, _ = ev.load_dataset("sorular_v5")
MOVED = {"D01", "D02", "D04", "A11", "A12", "A13", "A14", "P05"}


def test_only_expected_tools_and_group_of_moved_questions_change():
    old = {q["id"]: q for q in V5["sorular"]}
    new = {q["id"]: q for q in V6["sorular"]}
    assert set(old) == set(new)
    assert {i for i in old if old[i] != new[i]} == MOVED
    for i in MOVED:
        assert {k: v for k, v in old[i].items() if k not in ("beklenen_tool_lar", "grup")} == \
               {k: v for k, v in new[i].items() if k not in ("beklenen_tool_lar", "grup")}
    assert (V6["bugun"], V6["duyuru_deposu"]) == (V5["bugun"], V5["duyuru_deposu"])
