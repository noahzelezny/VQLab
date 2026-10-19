"""queue/steps: which step args are inputs and outputs, and the preflight redirect."""
from vqlab.queue import steps


def test_kl_pair_reads_the_per_position_dir_kl_ladder_writes(tmp_path):
    # 2026-10-18 F208 preflight: kl-pair's --per-pos-dir was taken for an
    # output, redirected to an empty folder, and never checked to exist
    assert steps.path_args(["--per-pos-dir", "/V/pp"], "kl-ladder") == ([], ["/V/pp"])
    assert steps.path_args(["--per-pos-dir", "/V/pp"], "kl-pair") == (["/V/pp"], [])
    om = {}
    steps.redirect_outs(["--per-pos-dir", "/V/pp"], tmp_path / "score", om, "kl-ladder")
    assert steps.redirect_outs(["--per-pos-dir", "/V/pp", "--arm", "a"],
                               tmp_path / "pair", om, "kl-pair") == \
        ["--per-pos-dir", str(tmp_path / "score" / "pp"), "--arm", "a"]
