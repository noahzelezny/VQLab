"""Family plugins feed the shared tables: a plugin's fit layout appears in
core.families.FAMILY, its scorer in stream_score.SCORERS, its tokenizer hook
in check-release's probe; families without a plugin get no hook."""
from vqlab.core import families
from vqlab.family import plugins, tokenizer_hook


def test_deepseek_plugin_wired():
    p = plugins()["deepseek_v4"]
    assert families.FAMILY["deepseek_v4"] is p.fit and p.fit["src_quant"] == "mxfp4"
    from vqlab.score import stream_score
    assert stream_score.SCORERS["deepseek_v4"]["fn"] is p.scorer["fn"]
    assert tokenizer_hook("deepseek_v4") is p.tokenizer_register
    assert tokenizer_hook("qwen3_5") is None
