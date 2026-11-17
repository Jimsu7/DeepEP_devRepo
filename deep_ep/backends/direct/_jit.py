"""Prepare only payload Dispatch for the selected transport before SHMEM collectives."""

from functools import lru_cache
import hashlib
from pathlib import Path
from threading import Lock


_INITIALIZATION_LOCK = Lock()
ABI = 2


def validate_native(module):
    """Reject old AOT or all-operator JIT extensions before device submission."""
    if getattr(module, "direct_jit_abi", None) != ABI or any(
        not callable(getattr(module, name, None)) for name in ("init_direct_jit", "prepare_direct_jit")
    ):
        raise ImportError("The installed Direct extension lacks Dispatch-only JIT ABI 2; rebuild the package")
    digest = getattr(module, "direct_jit_types_sha256", None)
    if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ImportError("The Direct extension has no valid tiling header identity; rebuild the package")


@lru_cache(maxsize=1)
def _library_root():
    return Path(__file__).resolve().parents[2]


def initialize(module, library_root):
    with _INITIALIZATION_LOCK:
        return _initialize(module, str(library_root))


@lru_cache(maxsize=None)
def _initialize(module, library_root):
    validate_native(module)
    root = Path(library_root).resolve()
    header = root / "jit/direct/common/types.h"
    if not header.is_file() or hashlib.sha256(header.read_bytes()).hexdigest() != module.direct_jit_types_sha256:
        raise RuntimeError("Direct JIT tiling header does not match the Host extension; rebuild the package")
    from deep_ep import _jit as shared
    from deep_ep._native import require_native

    # Both backends load the same bundled SHMEM. Hash the library actually
    # selected by the main native loader, not a different SDK installation.
    native = require_native("Direct JIT SHMEM dependency")
    includes, signature = shared._configuration(
        native, root, source_directory="jit/direct", include_directory="jit/direct"
    )
    module.init_direct_jit(str(root), includes, signature)


def prepare(module, transport):
    # Historical Direct aliases select the MTE kernel family. The native
    # resource layer independently validates the requested SHMEM engine.
    if transport not in ("auto", "mte", "roce", "sdma", "udma"):
        raise ValueError(f"Unsupported Direct JIT transport: {transport!r}")
    initialize(module, _library_root())
    # This remains context-aware in C++; do not memoize it by transport alone.
    module.prepare_direct_jit("udma" if transport == "udma" else "mte")


def prepare_collectively(module, transport, gather):
    """All ranks finish local preparation before anyone initializes SHMEM.

    The caller supplies its existing EP ProcessGroup/MPI gather, including on
    warm calls. Compilation/load failure must not strand peers in a barrier.
    """
    error = None
    try:
        prepare(module, transport)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    records = gather((transport, error))
    failures = [(rank, record[1]) for rank, record in enumerate(records) if record[1] is not None]
    if failures:
        message = "; ".join(f"rank {rank}: {detail}" for rank, detail in failures)
        raise RuntimeError(f"Direct JIT preparation failed collectively before SHMEM initialization: {message}")
    if any(record[0] != transport for record in records):
        raise ValueError("EP ranks selected different Direct JIT transports")
