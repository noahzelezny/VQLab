"""core/artifacts/artifact.py: the teacher files a build copies verbatim."""
from vqlab.core.artifacts.artifact import copy_config_files, copy_tokenizer_files


def test_teacher_tokenizer_files_are_copied_verbatim(tmp_path):
    """A re-serialized tokenizer.json is not the teacher's: transformers 5
    flipped the Qwen3.5 ByteLevel pre_tokenizer/decoder flags on save, and
    check-release failed the artifact against its own teacher."""
    src, dst = tmp_path / "teacher", tmp_path / "out"
    src.mkdir(), dst.mkdir()
    teacher = b'{"pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false}}'
    (src / "tokenizer.json").write_bytes(teacher)
    (src / "merges.txt").write_bytes(b"a b\n")
    (src / "config.json").write_text("{}")                   # not a tokenizer file
    resaved = b'{"pre_tokenizer": {"add_prefix_space": true}}'
    (dst / "tokenizer.json").write_bytes(resaved)
    (dst / "tokenizer_config.json").write_text("{}")        # teacher has none: kept
    assert copy_tokenizer_files(src, dst) == ["tokenizer.json", "merges.txt"]
    assert (dst / "tokenizer.json").read_bytes() == teacher
    assert (dst / "tokenizer_config.json").read_text() == "{}"
    assert not (dst / "config.json").exists()


def test_teacher_processor_config_is_copied(tmp_path):
    # GLM's struct base once dropped it; a vision build cannot serve images without it
    src, dst = tmp_path / "t", tmp_path / "o"
    src.mkdir()
    dst.mkdir()
    (src / "processor_config.json").write_text("{}")
    (src / "generation_config.json").write_text("{}")
    got = copy_config_files(src, dst)
    assert got == ["generation_config.json", "processor_config.json"]
    assert (dst / "processor_config.json").read_text() == "{}"


def test_a_graft_keeps_processor_configs_the_build_already_has(tmp_path):
    src, dst = tmp_path / "t", tmp_path / "o"
    src.mkdir()
    dst.mkdir()
    (src / "preprocessor_config.json").write_text("teacher")
    (src / "video_preprocessor_config.json").write_text("teacher")
    (dst / "preprocessor_config.json").write_text("staged")
    names = ("preprocessor_config.json", "video_preprocessor_config.json")
    assert copy_config_files(src, dst, names, overwrite=False) == [names[1]]
    assert (dst / "preprocessor_config.json").read_text() == "staged"


def test_a_single_file_checkpoint_is_indexed_from_its_header(tmp_path):
    # 2026-10-18: Qwen3-Embedding-0.6B ships one model.safetensors and no
    # index file; fit-dense died reading the index that small models lack
    import json
    import struct

    from vqlab.core.artifacts.artifact import Artifact
    hdr = json.dumps({"__metadata__": {"format": "pt"},
                      "a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
                      "b.weight": {"dtype": "F16", "shape": [1], "data_offsets": [2, 4]}}).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(hdr)) + hdr + b"\0" * 4)
    assert Artifact(tmp_path).index == {"a.weight": "model.safetensors",
                                        "b.weight": "model.safetensors"}

