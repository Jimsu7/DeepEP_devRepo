"""SDK-free checks for dependency hashing and warm package-path reuse."""

import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("dispatch_jit_signature_config", ROOT / "deep_ep/_jit.py")
jit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jit)


def legacy_signature(roots):
    digest = hashlib.sha256()
    for index, root in enumerate(roots):
        for path in sorted(root.rglob("*")):
            if path.is_file():
                name = f"{index}/{path.relative_to(root).as_posix()}".encode()
                data = path.read_bytes()
                digest.update(len(name).to_bytes(8, "little") + name)
                digest.update(len(data).to_bytes(8, "little") + data)
    return digest.hexdigest()


def tree(tmp_path):
    root = tmp_path / "sdk"
    (root / "a/sub").mkdir(parents=True)
    for relative, data in [
        ("a.txt", b"root"),
        ("a/z.h", b"header"),
        ("a/sub/x.hpp", b"nested"),
        (".hidden", b""),
        ("a-b", b"dash"),
    ]:
        (root / relative).write_bytes(data)
    return root


@pytest.mark.parametrize("layout", ["root", "nested-first", "parent-first", "duplicate"])
def test_signature_byte_stream_is_compatible(tmp_path, layout):
    root = tree(tmp_path)
    roots = {
        "root": [root],
        "nested-first": [root / "a", root, root / "a/sub"],
        "parent-first": [root, root / "a", root / "a/sub"],
        "duplicate": [root, root],
    }[layout]
    assert jit.content_signature(roots) == legacy_signature(roots)


def test_overlap_reads_each_file_once(tmp_path, monkeypatch):
    root = tree(tmp_path)
    roots = [root / "a", root, root / "a/sub", root]
    expected = legacy_signature(roots)
    counts = {}
    original = Path.read_bytes

    def read(path):
        counts[path] = counts.get(path, 0) + 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    assert jit.content_signature(roots) == expected
    assert len(counts) == 5 and set(counts.values()) == {1}


def test_same_size_same_timestamp_content_change_invalidates(tmp_path):
    root = tree(tmp_path)
    path = root / "a/z.h"
    original = path.stat()
    before = jit.content_signature([root, root / "a"])
    path.write_bytes(b"edited")
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert path.stat().st_size == original.st_size
    assert jit.content_signature([root, root / "a"]) != before


def test_file_add_remove_and_rename_invalidate(tmp_path):
    root = tree(tmp_path)
    before = jit.content_signature([root])
    added = root / "new.h"
    added.write_bytes(b"")
    after = jit.content_signature([root])
    assert after != before
    added.unlink()
    assert jit.content_signature([root]) == before
    (root / "a.txt").rename(root / "renamed.txt")
    assert jit.content_signature([root]) != before


def test_symlink_behavior_matches_original(tmp_path):
    root = tree(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "outside.h").write_bytes(b"not recursively included")
    try:
        (root / "alias.h").symlink_to(root / "a/z.h")
        (root / "linked-dir").symlink_to(outside, target_is_directory=True)
        (tmp_path / "root-link").symlink_to(root, target_is_directory=True)
        (root / "broken.h").symlink_to(root / "absent.h")
    except OSError:
        pytest.skip("symlink creation is unavailable")
    roots = [root, root / "a", tmp_path / "root-link", root / "linked-dir"]
    assert jit.content_signature(roots) == legacy_signature(roots)
    original = jit.content_signature([root])
    (outside / "outside.h").write_bytes(b"outside modification")
    assert jit.content_signature([root]) == original


def test_missing_and_non_directory_roots_fail(tmp_path):
    path = tmp_path / "file.h"
    path.write_text("header")
    for root in (path, tmp_path / "absent"):
        with pytest.raises(RuntimeError, match="missing"):
            jit.content_signature([root])


def test_prepare_resolves_package_only_once_and_retries_initialization(monkeypatch):
    jit._library_root.cache_clear()
    resolves = []
    original = Path.resolve

    def resolve(path, *args, **kwargs):
        resolves.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    calls = []

    def initialize(module, root):
        calls.append(root)
        if len(calls) == 1:
            raise RuntimeError("temporary SDK failure")

    monkeypatch.setattr(jit, "initialize", initialize)
    prepared = []
    module = SimpleNamespace(dispatch_jit_enabled=True, prepare_dispatch_jit=lambda *args: prepared.append(args))
    with pytest.raises(RuntimeError, match="temporary"):
        jit.prepare(module, 2, 128, 2, False, True)
    jit.prepare(module, 2, 128, 2, False, True)
    jit.prepare(module, 2, 128, 2, True, True)
    assert len(resolves) == 1
    assert len(calls) == 3 and len(prepared) == 2
    assert prepared[0] != prepared[1]
    jit._library_root.cache_clear()
