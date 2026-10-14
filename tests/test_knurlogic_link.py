"""gate/knurlogic_link.py: one directory symlink makes a scratch build
loadable by name, satisfying the gates' identity check; nothing else moves."""
import os

from vqlab.gate import knurlogic_link as kl


def _art(p):
    p.mkdir(parents=True)
    (p / "config.json").write_text("{}")
    return p


def _gate_accepts(md, art):
    # the check check-release --knurlogic-smoke and vision-smoke --knurlogic make
    name = art.resolve().name
    return (md / name).resolve() == art.resolve()


def test_links_a_scratch_build_and_the_gate_accepts_it(tmp_path, monkeypatch):
    md = tmp_path / "models"
    md.mkdir()
    monkeypatch.setenv("KNURLOGIC_MODELS", str(md))
    art = _art(tmp_path / "scratch" / "b" / "b-sz")
    assert not _gate_accepts(md, art)
    rc, msg = kl.link(art)
    assert rc == 0 and (md / "b-sz").is_symlink() and _gate_accepts(md, art)
    assert kl.link(art)[0] == 0                    # idempotent
    assert (art / "config.json").exists()


def test_refuses_what_it_would_have_to_remove(tmp_path, monkeypatch):
    md = tmp_path / "models"
    md.mkdir()
    monkeypatch.setenv("KNURLOGIC_MODELS", str(md))
    art = _art(tmp_path / "scratch" / "b-sz")
    real = _art(md / "b-sz")                        # a real folder of that name
    rc, msg = kl.link(art)
    assert rc == 2 and "real folder" in msg and real.is_dir() and not real.is_symlink()
    os.rename(real, tmp_path / "moved")
    other = _art(tmp_path / "elsewhere" / "b-sz")
    (md / "b-sz").symlink_to(other)
    rc, msg = kl.link(art)
    assert rc == 2 and "a link to" in msg and (md / "b-sz").resolve() == other.resolve()


def test_replaces_only_dangling_or_preflight_links(tmp_path, monkeypatch):
    md = tmp_path / "models"
    md.mkdir()
    monkeypatch.setenv("KNURLOGIC_MODELS", str(md))
    pf = tmp_path / "pf"
    monkeypatch.setenv("VQLAB_PREFLIGHT_DIR", str(pf))
    art = _art(tmp_path / "scratch" / "b-sz")
    (md / "b-sz").symlink_to(_art(pf / "q" / "skipzero" / "b-sz"))   # the preflight's copy
    assert kl.link(art)[0] == 0 and _gate_accepts(md, art)
    (md / "c").symlink_to(tmp_path / "gone")                        # dangling
    art2 = _art(tmp_path / "scratch" / "c")
    assert kl.link(art2)[0] == 0 and _gate_accepts(md, art2)


def test_refuses_a_missing_models_dir_or_a_non_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("KNURLOGIC_MODELS", str(tmp_path / "nowhere"))
    assert kl.link(_art(tmp_path / "a"))[0] == 2
    assert kl.link(tmp_path)[0] == 2
