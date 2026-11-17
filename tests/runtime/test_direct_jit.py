"""CPU-only source identity and collective admission for Direct Dispatch JIT."""

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def jit():
    return load("deep_ep/backends/direct/_jit.py", "isolated_direct_jit")


@pytest.fixture
def environment(tmp_path, monkeypatch, jit):
    root, cann, shmem = (tmp_path / name for name in ("package", "cann", "shmem"))
    for path in (
        root / "jit/kernels",
        root / "jit/direct/common",
        root / "jit/direct/kernels",
        cann / "include",
        cann / "x86_64-linux/include",
        cann / "compiler/tikcpp",
        cann / "asc/include/adv_api",
        shmem / "include",
        shmem / "src",
        root / ".libs",
    ):
        path.mkdir(parents=True)
        (path / "dependency.h").write_text("// dependency\n")
    header = root / "jit/direct/common/types.h"
    header.write_text("// native tiling identity\n")
    sdk_library = root / ".libs/libshmem.so"
    sdk_library.write_bytes(b"actual loaded SHMEM")
    native = ModuleType("fake_direct_native")
    native.direct_jit_abi = 2
    native.direct_jit_types_sha256 = hashlib.sha256(header.read_bytes()).hexdigest()
    native.initializations, native.preparations = [], []
    native.init_direct_jit = lambda *args: native.initializations.append(args)
    native.prepare_direct_jit = lambda transport: native.preparations.append(transport)
    shared = load("deep_ep/_jit.py", "isolated_mesh_jit")
    monkeypatch.setattr(shared.platform, "machine", lambda: "x86_64")
    package = ModuleType("deep_ep")
    package._jit = shared
    loader = ModuleType("deep_ep._native")
    core = ModuleType("fake_core_native")
    core.shmem_library_path = lambda: str(sdk_library)
    loader.require_native = lambda _: core
    monkeypatch.setitem(sys.modules, "deep_ep", package)
    monkeypatch.setitem(sys.modules, "deep_ep._native", loader)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(cann))
    monkeypatch.setenv("SHMEM_ROOT", str(shmem))
    monkeypatch.setattr(jit, "_library_root", lambda: root)
    return root, native, shared, core


def test_direct_preparation_initializes_once_but_checks_each_context(jit, environment):
    root, native, _, _ = environment
    for transport in ("mte", "mte", "udma"):
        jit.prepare(native, transport)
    assert len(native.initializations) == 1
    assert native.preparations == ["mte", "mte", "udma"]
    assert native.initializations[0][0] == str(root)
    includes = native.initializations[0][1]
    assert str(root / "jit/direct") in includes
    assert str(root / "jit") not in includes


@pytest.mark.parametrize("transport", ["auto", "mte", "roce", "sdma"])
def test_existing_direct_aliases_keep_mte_kernel_family(jit, environment, transport):
    _, native, _, _ = environment
    jit.prepare(native, transport)
    assert native.preparations == ["mte"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("direct_jit_abi", 0),
        ("direct_jit_abi", 1),
        ("direct_jit_types_sha256", "missing"),
        ("prepare_direct_jit", None),
    ],
)
def test_old_native_is_rejected_before_compile(jit, environment, field, value):
    _, native, _, _ = environment
    setattr(native, field, value)
    with pytest.raises(ImportError):
        jit.prepare(native, "mte")
    assert not native.initializations and not native.preparations


@pytest.mark.parametrize("missing", [False, True])
def test_native_source_abi_mismatch_fails_before_compile(jit, environment, missing):
    root, native, _, _ = environment
    header = root / "jit/direct/common/types.h"
    if missing:
        header.unlink()
    else:
        header.write_text("// wrong snapshot\n")
    with pytest.raises(RuntimeError, match="tiling header"):
        jit.prepare(native, "udma")
    assert not native.initializations and not native.preparations


def test_mesh_signature_does_not_enumerate_direct_sources(jit, environment):
    root, native, shared, core = environment
    mesh_before = shared._configuration(core, root)
    jit.initialize(native, root)
    direct_before = native.initializations[-1][-1]
    (root / "jit/direct/kernels/dispatch.cpp").write_text("// changed Direct kernel\n")
    assert shared._configuration(core, root) == mesh_before
    jit._initialize.cache_clear()
    jit.initialize(native, root)
    assert native.initializations[-1][-1] != direct_before


def test_direct_signature_uses_actual_bundled_shmem(jit, environment):
    root, native, _, _ = environment
    jit.initialize(native, root)
    before = native.initializations[-1][-1]
    (root / ".libs/libshmem.so").write_bytes(b"another loaded SHMEM")
    jit._initialize.cache_clear()
    jit.initialize(native, root)
    assert native.initializations[-1][-1] != before


def test_failed_initialization_is_retryable(jit, environment):
    _, native, _, _ = environment
    original = native.init_direct_jit

    def fail(*args):
        raise RuntimeError("temporary configure error")

    native.init_direct_jit = fail
    with pytest.raises(RuntimeError, match="temporary"):
        jit.prepare(native, "mte")
    native.init_direct_jit = original
    jit.prepare(native, "mte")
    assert len(native.initializations) == 1


def test_collective_preparation_reports_peer_failure_on_successful_rank(jit, monkeypatch):
    calls = []
    monkeypatch.setattr(jit, "prepare", lambda *args: calls.append(args))

    def gather(value):
        assert calls and value == ("udma", None)
        return [value, ("udma", "RuntimeError: Bisheng failed")]

    with pytest.raises(RuntimeError, match="rank 1.*Bisheng failed"):
        jit.prepare_collectively(object(), "udma", gather)


def test_collective_preparation_participates_after_local_failure(jit, monkeypatch):
    observed = []

    def fail(*args):
        raise RuntimeError("Bisheng failed")

    def gather(value):
        observed.append(value)
        return [("mte", None), value]

    monkeypatch.setattr(jit, "prepare", fail)
    with pytest.raises(RuntimeError, match="before SHMEM initialization.*rank 1"):
        jit.prepare_collectively(object(), "mte", gather)
    assert observed == [("mte", "RuntimeError: Bisheng failed")]


def test_collective_preparation_rejects_mixed_transports(jit, monkeypatch):
    monkeypatch.setattr(jit, "prepare", lambda *args: None)
    with pytest.raises(ValueError, match="different Direct JIT transports"):
        jit.prepare_collectively(object(), "mte", lambda value: [value, ("udma", None)])


def test_collective_warm_preparation_still_checks_all_ranks(jit, monkeypatch):
    observed = []
    monkeypatch.setattr(jit, "prepare", lambda *args: None)
    for _ in range(2):
        jit.prepare_collectively(object(), "mte", lambda value: observed.append(value) or [value, value])
    assert observed == [("mte", None), ("mte", None)]
