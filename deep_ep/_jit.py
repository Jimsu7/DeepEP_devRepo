# Copyright (c) 2026, Lu Lu
"""HT dispatch specialization and JIT configuration; no import-time NPU probing."""

import hashlib
from collections import Counter
from functools import lru_cache
import os
from pathlib import Path
import platform
import re
from threading import Lock

_initialization_lock = Lock()


def content_signature(roots):
    """Hash every dependency byte, sharing enumeration and reads for overlaps.

    The byte stream is unchanged from the original per-root rglob implementation,
    including root order and duplicate logical entries. Only work within this
    invocation is shared; timestamps never stand in for dependency contents.
    """
    roots = [Path(root) for root in roots]
    for root in roots:
        if not root.is_dir():
            raise RuntimeError(f"Dispatch JIT dependency directory is missing: {root}")
    canonical = [root.resolve() for root in roots]
    inventories = {}
    for root in sorted(set(canonical), key=lambda path: len(path.parts)):
        parent = next((path for path in inventories if root.is_relative_to(path)), None)
        if parent is not None:
            inventories[root] = tuple(path for path in inventories[parent] if path.is_relative_to(root))
            continue
        pending, files = [root], []
        while pending:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    # rglob does not descend through directory symlinks. A file
                    # symlink does contribute its target bytes under its own name.
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    elif entry.is_file():
                        files.append(Path(entry.path))
        inventories[root] = tuple(sorted(files))
    remaining = Counter(path for root in canonical for path in inventories[root])
    shared = {}
    digest = hashlib.sha256()
    for index, root in enumerate(canonical):
        prefix_length = len(root.parts)
        for path in inventories[root]:
            # Inventory membership already establishes this root prefix. Taking
            # the suffix avoids pathlib's repeated ancestor walk per entry.
            relative_name = "/".join(path.parts[prefix_length:])
            name = f"{index}/{relative_name}".encode()
            data = shared.get(path)
            if data is None:
                data = path.read_bytes()
                if remaining[path] > 1:
                    shared[path] = data
            digest.update(len(name).to_bytes(8, "little"))
            digest.update(name)
            digest.update(len(data).to_bytes(8, "little"))
            digest.update(data)
            remaining[path] -= 1
            if not remaining[path]:
                shared.pop(path, None)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _library_root():
    """Resolve the installed package once, without import-time filesystem work."""
    return Path(__file__).resolve().parent


def specialization(world, hidden, topk, fp8, with_weights, profile=False):
    """Return the complete compile-time identity; tokens/rank/rows stay dynamic."""
    if type(world) is not int or world < 2 or world > 512 or world & (world - 1):
        raise ValueError("Dispatch JIT world must be a power of two from 2 to 512")
    if any(type(value) is not bool for value in (fp8, with_weights, profile)):
        raise TypeError("Dispatch JIT fp8, with_weights and profile must be bool")
    if type(hidden) is not int or not 0 < hidden <= 7168 or hidden % (128 if fp8 else 16):
        raise ValueError("Dispatch JIT hidden must be <=7168 and aligned to BF16:16/FP8:128")
    if type(topk) is not int or not 1 <= topk <= 16:
        raise ValueError("Dispatch JIT topk must be 1..16")
    return world, hidden, topk, fp8, with_weights, profile


def initialize(module, library_root):
    with _initialization_lock:
        return _initialize(module, str(library_root))


@lru_cache(maxsize=None)
def _initialize(module, library_root):
    library_root = Path(library_root).resolve()
    cann_root = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    shmem_root = os.environ.get("SHMEM_ROOT")
    if not cann_root or not shmem_root:
        raise RuntimeError("Dispatch JIT requires ASCEND_HOME_PATH (or ASCEND_TOOLKIT_HOME) and SHMEM_ROOT")
    cann, shmem = Path(cann_root).resolve(), Path(shmem_root).resolve()
    tikcpp = cann / "compiler/tikcpp"
    arch_include = cann / f"{platform.machine()}-linux/include"
    includes = [
        library_root / "jit",
        cann / "include",
        arch_include,
        tikcpp,
        tikcpp / "tikcfw",
        tikcpp / "tikcfw/impl",
        tikcpp / "tikcfw/interface",
        shmem / "include",
        shmem / "src",
        shmem / "src/device",
        shmem / "src/device_simt",
    ]
    rt_candidates = [
        cann / "include/runtime",
        cann / "include/experiment/runtime",
        cann / "include",
        cann / "include/acl/runtime",
        cann / "runtime/include",
        arch_include / "runtime",
        arch_include / "experiment/runtime",
        arch_include / "acl/runtime",
        arch_include,
        cann / "pkg_inc/runtime",
        cann / f"{platform.machine()}-linux/pkg_inc/runtime",
        cann / f"{platform.machine()}-linux/runtime/include",
    ]
    compatibility = next((path for path in rt_candidates if (path / "rt.h").is_file()), None)
    if compatibility is not None and compatibility not in includes:
        includes.append(compatibility)
    includes = [path for path in includes if path.is_dir()]
    # DeepJIT's upstream compiler command concatenates paths without quoting.
    for path in [library_root, cann, shmem, *includes]:
        if not re.fullmatch(r"[A-Za-z0-9_./:+-]+", str(path)):
            raise RuntimeError(f"Dispatch JIT paths contain whitespace or shell metacharacters: {path}")
    asc_candidates = [
        cann / f"{platform.machine()}-linux/asc/include/adv_api",
        cann / "asc/include/adv_api",
        cann / "aarch64-linux/asc/include/adv_api",
    ]
    asc_include = next((path for path in asc_candidates if path.is_dir()), None)
    if asc_include is None:
        raise RuntimeError(f"Ascend adv_api include directory is missing under {cann}")
    roots = [
        library_root / "jit/kernels",
        shmem / "include",
        shmem / "src",
        cann / "include",
        arch_include,
        tikcpp,
        asc_include,
    ]
    if compatibility is not None and not any(compatibility.is_relative_to(root) for root in roots):
        roots.append(compatibility)
    signature = content_signature(roots)
    library = (
        Path(module.shmem_library_path()).resolve()
        if hasattr(module, "shmem_library_path")
        else shmem / "lib/libshmem.so"
    )
    if not library.is_file():
        raise RuntimeError(f"Dispatch JIT SHMEM library is missing: {library}")
    signature = hashlib.sha256((signature + hashlib.sha256(library.read_bytes()).hexdigest()).encode()).hexdigest()
    module.init_dispatch_jit(str(library_root), [str(path) for path in includes], signature)


def prepare(module, world, hidden, topk, fp8, with_weights, profile=False):
    spec = specialization(world, hidden, topk, fp8, with_weights, profile)
    if not getattr(module, "dispatch_jit_enabled", False):
        raise RuntimeError("The installed DeepEP extension lacks HT dispatch JIT; rebuild the package")
    initialize(module, _library_root())
    return module.prepare_dispatch_jit(*spec)
