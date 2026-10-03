"""card-tables renders exactly what the scorer's JSON says."""
import json

from vqlab.records import card_tables as CT


def test_render(tmp_path, capsys):
    t = {c: {"mean_kl_millinats": v} for c, v in zip(("prose", "code", "lit"), (30.0, 9.0, 3.0))}
    t2 = {c: {"mean_kl_millinats": v} for c, v in zip(("prose", "code", "lit"), (60.0, 18.0, 6.0))}
    j = tmp_path / "k.json"
    j.write_text(json.dumps({"table": {"a": t, "b": t2}, "rungs": {"a": "/nope", "b": "/nope"},
                             "reference": "b", "paired": {"a": {c: {"delta": -1.0, "t": -3.0}
                                                                for c in ("prose", "code", "lit")}}}))
    CT.main(["--kl", str(j), "--this", "a", "--label", "a=Rel"])
    out = capsys.readouterr().out
    assert "| **Rel** | **?** | **30.0** | **9.0** | **3.0** | **14.0** |" in out
    assert "| b | ? | 60.0 | 18.0 | 6.0 | 28.0 |" in out
    assert out.index("| b |") < out.index("**Rel**")          # worst first
    assert "-1.0 (t -3.0)" in out
