"""SDK-free configuration, specialization identity and validation-tool checks.

These checks do not establish Ascend ABI, ELF loading or multi-rank correctness.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("dispatch_jit_config", ROOT / "deep_ep/_jit.py")
jit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jit)


def test_signature_tracks_transitive_source_sdk_and_relocation(tmp_path):
    kernel, sdk = tmp_path / "kernel", tmp_path / "sdk"
    kernel.mkdir()
    sdk.mkdir()
    (kernel / "entry.hpp").write_text('#include "detail.hpp"\n')
    detail, sdk_header = kernel / "detail.hpp", sdk / "shmem.h"
    detail.write_text("constexpr int value = 1;\n")
    sdk_header.write_text("// SDK one\n")
    original = jit.content_signature([kernel, sdk])
    detail.write_text("constexpr int value = 2;\n")
    changed = jit.content_signature([kernel, sdk])
    assert changed != original
    sdk_header.write_text("// SDK two\n")
    final = jit.content_signature([kernel, sdk])
    assert final != changed
    moved = tmp_path / "moved"
    kernel.rename(moved)
    assert jit.content_signature([moved, sdk]) == final
    import shutil

    relocated = tmp_path / "relocated"
    shutil.copytree(moved, relocated)
    assert jit.content_signature([moved]) == jit.content_signature([relocated])


def test_missing_dependencies_fail_before_native_compile(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="missing"):
        jit.content_signature([tmp_path / "missing"])
    for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "SHMEM_ROOT"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="requires"):
        jit.initialize(object(), tmp_path)


@pytest.mark.parametrize("layout", ["x86_64-linux", "aarch64-linux", "neutral"])
def test_jit_finds_same_sdk_adv_api_layout(tmp_path, monkeypatch, layout):
    cann, shmem, library = (tmp_path / name for name in ("cann", "shmem", "library"))
    for path in (
        cann / "include",
        cann / "x86_64-linux/include",
        cann / "compiler/tikcpp",
        shmem / "include",
        shmem / "src",
        library / "jit/kernels",
    ):
        path.mkdir(parents=True, exist_ok=True)
        (path / "header.h").write_text("// dependency\n")
    adv = cann / ("asc/include/adv_api" if layout == "neutral" else f"{layout}/asc/include/adv_api")
    adv.mkdir(parents=True)
    (adv / "device.h").write_text("// adv_api\n")
    (shmem / "lib").mkdir()
    (shmem / "lib/libshmem.so").write_bytes(b"test SDK")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(cann))
    monkeypatch.setenv("SHMEM_ROOT", str(shmem))
    monkeypatch.setattr(jit.platform, "machine", lambda: "x86_64")
    observed = []
    module = type("Native", (), {"init_dispatch_jit": staticmethod(lambda *args: observed.append(args))})()
    jit.initialize(module, library)
    assert len(observed) == 1
    first_signature = observed[0][-1]
    (adv / "device.h").write_text("// updated adv_api\n")
    jit._initialize.cache_clear()
    jit.initialize(module, library)
    assert observed[-1][-1] != first_signature


def test_complete_six_field_specialization_identity():
    baseline = (64, 7168, 8, False, True, False)
    assert jit.specialization(*baseline) == baseline
    variants = [(i, value) for i, value in enumerate((128, 4096, 6, True, False, True))]
    keys = {baseline}
    for index, value in variants:
        changed = list(baseline)
        changed[index] = value
        keys.add(jit.specialization(*changed))
    assert len(keys) == 7  # Every field participates, including Profile and WithWeights.


@pytest.mark.parametrize(
    "values",
    [
        (3, 7168, 8, False, True, False),
        (1024, 7168, 8, False, True, False),
        (64, 7200, 8, False, True, False),
        (64, 4112, 8, True, True, False),
        (64, 4096, 0, False, True, False),
        (64, 4096, 17, False, True, False),
        (64, 4096, 8, 1, True, False),
    ],
)
def test_invalid_specialization_rejected(values):
    with pytest.raises((ValueError, TypeError)):
        jit.specialization(*values)


def test_prepare_transmits_exact_validated_spec(monkeypatch):
    observed = []
    module = SimpleNamespace(dispatch_jit_enabled=True, prepare_dispatch_jit=lambda *values: observed.append(values))
    monkeypatch.setattr(jit, "initialize", lambda native, root: None)
    jit.prepare(module, 128, 4096, 6, True, False, True)
    assert observed == [(128, 4096, 6, True, False, True)]


def test_prepare_rejects_an_old_installed_extension(monkeypatch):
    monkeypatch.setattr(
        jit, "initialize", lambda native, root: pytest.fail("old extension must fail before SDK discovery")
    )
    with pytest.raises(RuntimeError, match="rebuild the package"):
        jit.prepare(object(), 128, 4096, 6, True, False, False)


def test_validator_dynamic_data_does_not_change_specialization_key():
    from scripts.jit.validate_dispatch import specialization_key

    case = dict(
        tokens=33,
        hidden=4096,
        topk=8,
        experts=1024,
        dtype="fp8",
        weights=True,
        profile=False,
        route="random",
        seed=1,
        routing="fresh",
        dedup_mode="server",
        benchmark_barriers=False,
    )
    key = specialization_key(16, case)
    assert key == jit.specialization(16, 4096, 8, True, True, False)
    dynamic = case | dict(
        tokens=256,
        experts=512,
        route="duplicates",
        seed=9,
        routing="cached",
        dedup_mode="rank",
        benchmark_barriers=True,
    )
    assert specialization_key(16, dynamic) == key
    assert specialization_key(16, case | dict(hidden=7168)) != key


@pytest.mark.parametrize("machine_type", ["mesh_clos", "global_clos", "mesh_pcie"])
def test_validator_qualifies_only_the_selected_jit_backend(monkeypatch, machine_type):
    from deep_ep import _machine
    from scripts.jit.validate_dispatch import select_jit_backend

    group, observed = object(), []
    selection = dict(machine_type=machine_type, source="rootinfo")

    def select(**kwargs):
        observed.append(kwargs)
        return selection

    monkeypatch.setattr(_machine, "select", select)
    if machine_type == "mesh_clos":
        assert select_jit_backend(group) is selection
    else:
        with pytest.raises(NotImplementedError, match=f"selected {machine_type} uses Direct kernels"):
            select_jit_backend(group)
    assert observed == [dict(group=group)]


def test_validator_matrix_revisits_first_spec_and_preserves_profile(tmp_path):
    from scripts.jit.validate_dispatch import load_cases, specialization_key

    path = tmp_path / "cases.json"
    path.write_text(json.dumps([dict(hidden=4096), dict(topk=4, weights=False, profile=True)]))
    args = SimpleNamespace(
        cases=path,
        ep_size=2,
        tokens=33,
        hidden=7168,
        topk=6,
        experts=1024,
        dtype="bf16",
        weights=True,
        profile=False,
        route="random",
        seed=1,
        routing="alternating",
        dedup_mode="auto",
        benchmark_barriers=False,
    )
    cases = load_cases(args)
    assert len(cases) == 3 and cases[0] == cases[-1]
    assert specialization_key(2, cases[0]) != specialization_key(2, cases[1])
    assert cases[1]["profile"] is True
    path.write_text('[{"hidden": 4112, "dtype": "fp8"}]')
    with pytest.raises(ValueError, match="aligned"):
        load_cases(args)


def test_validation_uses_medians_and_propagates_peer_failure():
    from scripts.jit.validate_dispatch import summarize, check_all_ranks

    samples = [dict(wall_us=x, kernel_us=x / 2, device_total_us=x * 2) for x in (1, 100, 3)]
    assert summarize(samples)["wall_us"] == 3

    class Group:
        def all_gather_object(self, output, value):
            output[:] = [value, "peer failed"]

    with pytest.raises(AssertionError, match="peer failed"):
        check_all_ranks(Group(), 2, lambda: None)


def test_cache_matrix_reuses_loaded_handle_when_revisiting_spec():
    from scripts.jit.validate_dispatch import validate_cache_record

    first, second = (128, 7168, 8, False, True, False), (128, 4096, 6, False, True, False)
    history = {}
    # Unrelated preloaded kernels are permitted; exact count growth is not assumed.
    count = validate_cache_record(history, first, dict(key="A", context_kernel_count=4, handle_reused=False))
    count = validate_cache_record(history, second, dict(key="B", context_kernel_count=5, handle_reused=False), count)
    assert validate_cache_record(history, first, dict(key="A", context_kernel_count=5, handle_reused=True), count) == 5
    with pytest.raises(AssertionError, match="reuse its loaded handle"):
        validate_cache_record(history, first, dict(key="A", context_kernel_count=5, handle_reused=False), count)
    with pytest.raises(AssertionError, match="changed the context kernel count"):
        validate_cache_record(history, first, dict(key="A", context_kernel_count=6, handle_reused=True), count)


def test_cache_records_reject_artifact_collision_and_dynamic_handle_growth():
    from scripts.jit.validate_dispatch import validate_cache_record, _same_key

    history = {(128, 7168, 8, False, True, False): "A"}
    with pytest.raises(AssertionError, match="shared one artifact"):
        validate_cache_record(
            history, (128, 4096, 6, False, True, False), dict(key="A", context_kernel_count=2, handle_reused=False), 1
        )
    with pytest.raises(AssertionError, match="distinct context handles"):
        validate_cache_record(
            history, (128, 4096, 6, False, True, False), dict(key="B", context_kernel_count=1, handle_reused=False), 1
        )
    before = dict(key="A", context_kernel_count=1, handle_reused=False)
    _same_key(before, dict(key="A", context_kernel_count=1, handle_reused=True))
    with pytest.raises(AssertionError, match="loaded handle to be replaced"):
        _same_key(before, before)
    with pytest.raises(AssertionError, match="increased the context kernel count"):
        _same_key(before, dict(key="A", context_kernel_count=2, handle_reused=True))


def test_validator_preserves_latest_rank_and_server_plans():
    from scripts.jit.validate_dispatch import rank_dedup_enabled

    for world in (8, 16):
        assert rank_dedup_enabled(world, "auto") is True
        assert rank_dedup_enabled(world, "rank") is True
        assert rank_dedup_enabled(world, "server") is False
    assert rank_dedup_enabled(128, "auto") is False
    with pytest.raises(ValueError, match="EP8/16"):
        rank_dedup_enabled(32, "rank")
    with pytest.raises(ValueError, match="auto, rank or server"):
        rank_dedup_enabled(8, "invalid")


def test_validator_uses_named_device_total_without_implied_barriers():
    from scripts.jit.validate_dispatch import device_timing

    # Reordering the named records must not silently change what is called kernel time.
    result = device_timing(
        dict(
            device_phase_names=("kernel", "entry_barrier", "device_total", "exit_barrier"),
            device_ms=(1.0, 0.01, 1.03, 0.02),
        )
    )
    assert result["kernel_us"] == 1000 and result["device_total_us"] == 1030
    assert result["device_phase_us"]["entry_barrier"] == 10
    with pytest.raises(ValueError, match="four named"):
        device_timing(dict(device_phase_names=("kernel",), device_ms=(1.0,)))
    with pytest.raises(ValueError, match="identify kernel"):
        device_timing(dict(device_phase_names=("a", "b", "c", "d"), device_ms=(1.0,) * 4))


def test_jit_compile_preflight_is_forced_without_global_debug(monkeypatch):
    """Run the production function with a CPU-only collective transport stub."""
    import ast
    from contextlib import nullcontext
    import sys
    from types import ModuleType

    path = ROOT / "deep_ep/backends/mesh_clos/elastic.py"
    node = next(
        item
        for item in ast.parse(path.read_text()).body
        if isinstance(item, ast.FunctionDef) and item.name == "_collective_check"
    )
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    check = namespace["_collective_check"]
    records, failed_rank = [], None

    class Tensor:
        def __init__(self, values, device="cpu", dtype=None):
            self.values, self.device, self.dtype = list(values), device, dtype

    class Matrix:
        def __init__(self, shape, **kwargs):
            self.rows = [Tensor([0] * shape[1], **kwargs) for _ in range(shape[0])]

        def unbind(self, axis):
            assert axis == 0
            return self.rows

        def cpu(self):
            return self

        def tolist(self):
            return [row.values for row in self.rows]

    torch, dist = ModuleType("torch"), ModuleType("torch.distributed")
    torch.distributed, torch.int64 = dist, object()
    torch.inference_mode = lambda *_: nullcontext()
    torch.no_grad = nullcontext
    torch.tensor = lambda values, **kwargs: Tensor(values, **kwargs)
    torch.empty = lambda shape, **kwargs: Matrix(shape, **kwargs)
    dist.get_backend = lambda group: "gloo"

    def gather(peers, local, group):
        assert len(local.values) == 18
        records.append(list(local.values))
        for rank, peer in enumerate(peers):
            peer.values[:] = local.values
            if rank == failed_rank:
                peer.values[0] = 1

    dist.all_gather = gather
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    runtime = SimpleNamespace(debug_collectives=False, group=object(), world_size=2, device=0)
    check(runtime, None, [0] * 9, "Notify")
    check(runtime, None, [0] * 9, "Combine")
    assert not records
    with pytest.raises(ValueError, match="local failure"):
        check(runtime, ValueError("local failure"), [], "Notify")
    check(runtime, None, [0] * 9, "Dispatch", phase=0, force=True)
    check(runtime, None, [0] * 14, "Dispatch", phase=1, force=True)
    assert [record[1:4] for record in records] == [[1, 0, 9], [1, 1, 14]]
    failed_rank = 1
    with pytest.raises(ValueError, match=r"Dispatch preflight failed on ranks \[1\]"):
        check(runtime, None, [0] * 14, "Dispatch", phase=1, force=True)
    failed_rank = None
    with pytest.raises(ValueError, match="signature exceeds 14"):
        check(runtime, None, [0] * 15, "Dispatch", phase=1, force=True)
