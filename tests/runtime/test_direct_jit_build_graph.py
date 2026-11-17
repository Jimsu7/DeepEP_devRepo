"""Configure the real device build graph without compiling or loading NPU code.

The host compiler is used only for CMake's compiler-identification probes. Fake
SDK files satisfy find_library; no target is built or installed by these tests.
"""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_AOT = {
    "csrc/kernels/notify/notify.cpp",
    "csrc/kernels/combine/combine.cpp",
    "csrc/kernels/combine/reuse.cpp",
    "csrc/kernels/combine/combine_ep8.cpp",
    "csrc/kernels/dispatch/low_latency.cpp",
    "csrc/kernels/combine/low_latency.cpp",
    "csrc/direct/kernels/abi.cpp",
    "csrc/direct/kernels/dispatch/dispatch_layout.cpp",
    "csrc/direct/kernels/notify/notify_dispatch.cpp",
    "csrc/direct/kernels/notify/notify_combine.cpp",
    "csrc/direct/kernels/combine/combine.cpp",
    "csrc/direct/kernels/combine/combine_udma.cpp",
    "csrc/direct/kernels/expand/expand.cpp",
    "csrc/direct/kernels/aggregate/aggregate.cpp",
}
JIT_ONLY = {
    "csrc/kernels/dispatch/dispatch.cpp",
    "csrc/direct/kernels/dispatch/dispatch.cpp",
    "csrc/direct/kernels/dispatch/dispatch_udma.cpp",
}


@pytest.fixture(scope="module")
def configured(tmp_path_factory):
    if sys.platform != "linux":
        pytest.skip("The production device CMake graph requires Linux")
    cmake = os.environ.get("CMAKE_COMMAND") or shutil.which("cmake")
    compiler = shutil.which("g++") or shutil.which("clang++")
    generator = "Ninja" if shutil.which("ninja") else "Unix Makefiles" if shutil.which("make") else None
    if not cmake or not compiler or not generator:
        pytest.skip("Requires real CMake, a host compiler, and Ninja or Make")
    temporary = tmp_path_factory.mktemp("device-jit-build-graph")
    cann, shmem = temporary / "CANN SDK", temporary / "SHMEM SDK"
    for path in (
        cann / "lib64/libascendcl.so",
        cann / "lib64/libruntime.so",
        cann / "include/acl/acl.h",
        shmem / "lib/libshmem.so",
        shmem / "include/shmem.h",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"Configuration-only fixture; never load or link this file.\n")
    child = temporary / "device-build"
    query = child / ".cmake/api/v1/query"
    query.mkdir(parents=True)
    (query / "codemodel-v2").touch()

    def configure(source, binary):
        command = [
            str(cmake),
            "-S",
            str(source),
            "-B",
            str(binary),
            "-G",
            generator,
            "-DCMAKE_CXX_COMPILER=" + str(compiler),
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DDEEPEP_SOURCE_DIR=" + str(ROOT),
            "-DASCEND_HOME_PATH=" + str(cann),
            "-DSHMEM_ROOT=" + str(shmem),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    configure(ROOT / "cmake/device", child)

    # Exercise the actual parent Device.cmake as well, without executing its
    # custom target. This checks imported targets, byproducts and install rules.
    parent_source = temporary / "parent-harness"
    parent_source.mkdir()
    (parent_source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(device_graph_probe LANGUAGES CXX)\n"
        f'include("{(ROOT / "cmake/Device.cmake").as_posix()}")\n'
        "get_property(imported DIRECTORY PROPERTY IMPORTED_TARGETS)\n"
        "get_property(build_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)\n"
        'file(WRITE "${CMAKE_BINARY_DIR}/targets.txt" "${imported};${build_targets}")\n'
        "get_target_property(location deepep_device IMPORTED_LOCATION)\n"
        'file(WRITE "${CMAKE_BINARY_DIR}/location.txt" "${location}")\n'
        "get_target_property(direct_location deepep_clos_device IMPORTED_LOCATION)\n"
        'file(WRITE "${CMAKE_BINARY_DIR}/direct-location.txt" "${direct_location}")\n',
        encoding="utf-8",
    )
    parent = temporary / "parent-build"
    configure(parent_source, parent)
    return child, parent


def test_actual_compile_commands_restore_non_dispatch_aot_and_exclude_ht_dispatch(configured):
    child, _ = configured
    commands = json.loads((child / "compile_commands.json").read_text())
    sources = set()
    unity_groups = []
    for item in commands:
        source = Path(item["file"]).resolve()
        if source.is_relative_to(ROOT):
            sources.add(source.relative_to(ROOT).as_posix())
        else:
            # CMake unity sources include the original files in one translation
            # unit. Expand that generated unit before checking what is compiled.
            assert source.is_relative_to(child) and source.name.startswith("unity_")
            includes = re.findall(r'^#include "([^"]+)"', source.read_text(), re.MULTILINE)
            assert includes
            group = [Path(path).resolve().relative_to(ROOT).as_posix() for path in includes]
            sources.update(group)
            unity_groups.append(group)
    assert sources == EXPECTED_AOT
    assert not sources & JIT_ONLY
    assert len(commands) == 10  # Six Mesh units; Direct ABI, two Combine units and one unity unit.
    assert len(unity_groups) == 1
    group = unity_groups[0]
    assert set(group) == {
        "csrc/direct/kernels/dispatch/dispatch_layout.cpp",
        "csrc/direct/kernels/notify/notify_dispatch.cpp",
        "csrc/direct/kernels/notify/notify_combine.cpp",
        "csrc/direct/kernels/expand/expand.cpp",
        "csrc/direct/kernels/aggregate/aggregate.cpp",
    }
    # Expand/Aggregate inherit SHMEM declarations through Notify in the original
    # unity translation unit. Restoring AOT must retain this SDK include order.
    notify = group.index("csrc/direct/kernels/notify/notify_dispatch.cpp")
    assert notify < group.index("csrc/direct/kernels/expand/expand.cpp")
    assert notify < group.index("csrc/direct/kernels/aggregate/aggregate.cpp")
    for item in commands:
        command = item.get("command", " ".join(item.get("arguments", [])))
        assert "--npu-arch=dav-3510" in command and "-xasc" in command
    assert not list((child / "CMakeFiles").rglob("*.o")), "Configure-only audit unexpectedly compiled a target object"
    assert not list(child.glob("*.so")), "Configure-only audit unexpectedly linked a device library"


def test_actual_device_codemodel_has_mesh_and_non_dispatch_direct_targets(configured):
    child, _ = configured
    reply = child / ".cmake/api/v1/reply"
    index = json.loads(next(reply.glob("index-*.json")).read_text())
    codemodel = json.loads((reply / index["reply"]["codemodel-v2"]["jsonFile"]).read_text())
    targets = [target for config in codemodel["configurations"] for target in config["targets"]]
    assert {target["name"] for target in targets} == {"deepep_kernels", "deepep_clos_kernels"}
    for target in targets:
        generated = json.loads((reply / target["jsonFile"]).read_text())
        assert {artifact["path"] for artifact in generated["artifacts"]} == {f"lib{target['name']}.so"}
        generated_sources = {
            Path(source["path"]).as_posix() for source in generated["sources"] if not source.get("isGenerated", False)
        }
        assert not any(source.endswith(tuple(JIT_ONLY)) for source in generated_sources)


def test_actual_parent_import_and_install_rules_include_non_dispatch_direct_library(configured):
    _, parent = configured
    targets = set((parent / "targets.txt").read_text().split(";"))
    assert targets == {"deepep_device", "deepep_clos_device", "device_build"}
    assert (parent / "location.txt").read_text().endswith("/device/libdeepep_kernels.so")
    assert (parent / "direct-location.txt").read_text().endswith("/device/libdeepep_clos_kernels.so")
    install = (parent / "cmake_install.cmake").read_text()
    assert "libdeepep_kernels.so" in install
    assert "libdeepep_clos_kernels.so" in install
    generated_build = "\n".join(
        generated.read_text() for generated in list(parent.rglob("*.make")) + list(parent.glob("*.ninja"))
    )
    assert "libdeepep_clos_kernels.so" in generated_build
