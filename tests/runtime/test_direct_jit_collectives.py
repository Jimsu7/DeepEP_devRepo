"""Real CPU Gloo checks that Direct preparation cannot strand SHMEM peers."""

from datetime import timedelta
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest


def _worker(rank, store, directory):
    import torch.distributed as dist
    from deep_ep.backends.direct import buffer, _jit

    dist.init_process_group("gloo", init_method=store, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    reports = []
    calls = {"uid": 0, "constructor": 0, "class_lookup": 0}

    def uid():
        calls["uid"] += 1
        return [7, 8, 9]

    native = SimpleNamespace(get_shmem_bootstrap=lambda _: (0, []), get_shmem_unique_id=uid)

    class Runtime:
        def __init__(self, *args):
            calls["constructor"] += 1

    def native_class(name):
        calls["class_lookup"] += 1
        assert name == "BufferRuntime"
        return Runtime

    def construct():
        return buffer.Buffer(dist.group.WORLD, num_bytes=2 * 1024 * 1024, transport="udma", explicitly_destroy=True)

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(buffer, "_is_capturing", lambda: False)
            patch.setattr(buffer, "_create_communication_stream", lambda: object())
            patch.setattr(buffer, "get_native_class", native_class)
            patch.setattr(buffer, "load_native_module", lambda: native)

            def compile_failure(*args):
                if rank == 1:
                    raise RuntimeError("simulated Bisheng error")

            patch.setattr(_jit, "prepare", compile_failure)
            with pytest.raises(RuntimeError, match="failed collectively.*rank 1.*simulated Bisheng"):
                construct()
            assert calls == {"uid": 0, "constructor": 0, "class_lookup": 0}
            reports.append("compiler_failure_blocks_shmem_on_both_ranks")

            def load_failure():
                if rank == 1:
                    raise ImportError("old Direct JIT ABI")
                return native

            patch.setattr(buffer, "load_native_module", load_failure)
            patch.setattr(_jit, "prepare", lambda *args: pytest.fail("prepared after peer loader failure"))
            with pytest.raises(RuntimeError, match="native load failed collectively.*rank 1.*old Direct JIT ABI"):
                construct()
            assert calls == {"uid": 0, "constructor": 0, "class_lookup": 0}
            reports.append("loader_failure_blocks_shmem_on_both_ranks")

            patch.setattr(buffer, "load_native_module", lambda: native)
            patch.setattr(_jit, "prepare", lambda *args: None)
            instance = construct()
            assert calls == {"uid": int(rank == 0), "constructor": 1, "class_lookup": 1}
            assert instance.transport == "udma"
            reports.append("retry_preserves_collective_order")
        Path(directory, f"rank{rank}.json").write_text(json.dumps(reports))
    finally:
        dist.destroy_process_group()


def test_direct_jit_failure_admission_with_real_cpu_group(tmp_path):
    dist = pytest.importorskip("torch.distributed")
    if not dist.is_gloo_available():
        pytest.skip("Requires CPU Gloo")
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    workers = [
        context.Process(target=_worker, args=(rank, (tmp_path / "store").resolve().as_uri(), str(tmp_path)))
        for rank in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + 90
        for worker in workers:
            worker.join(max(0, deadline - time.monotonic()))
        assert [worker.exitcode for worker in workers] == [0, 0]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            if worker.pid is not None:
                worker.join(5)
                if worker.is_alive():
                    worker.kill()
                    worker.join(5)
    expected = [
        "compiler_failure_blocks_shmem_on_both_ranks",
        "loader_failure_blocks_shmem_on_both_ranks",
        "retry_preserves_collective_order",
    ]
    for rank in range(2):
        assert json.loads((tmp_path / f"rank{rank}.json").read_text()) == expected
