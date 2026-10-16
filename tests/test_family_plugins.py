"""Family plugins feed the shared tables: a plugin's fit layout appears in
family.registry.FAMILY, its scorer in stream_score.SCORERS, its tokenizer hook
in check-release's probe; families without a plugin get no hook."""
from vqlab.family import registry as families
from vqlab.family import plugins, tokenizer_hook


def test_deepseek_plugin_wired():
    p = plugins()["deepseek_v4"]
    assert families.FAMILY["deepseek_v4"] is p.fit and p.fit["src_quant"] == "mxfp4"
    from vqlab.score import stream_score
    assert stream_score.SCORERS["deepseek_v4"]["fn"] is p.scorer["fn"]
    assert tokenizer_hook("deepseek_v4") is p.tokenizer_register
    assert tokenizer_hook("qwen3_5") is None


def test_keep_fp_is_a_tuple_of_suffixes():
    """keep_fp feeds str.endswith, which also takes a bare str and would then
    match single characters without an error."""
    for p in plugins().values():
        assert isinstance(p.keep_fp, tuple) and p.keep_fp, p.name
        assert all(isinstance(s, str) and s for s in p.keep_fp), p.name
