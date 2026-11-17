"""SDK-free Direct JIT registry and real CPU-preprocessor entry isolation checks.

The preprocessing uses empty SDK header stubs. It proves which original device
entry bodies reach Bisheng, not that CANN compiles or executes those bodies.
"""

import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
DIRECT = ROOT / "csrc/direct"
KERNELS = sorted((DIRECT / "kernels").rglob("*.cpp"))
GLOBAL = re.compile(r"__global__\s+__aicore__\s+void\s+(\w+)\s*\(")

# Baseline 7878654: only payload Dispatch gains selection guards. A deliberate
# algorithm change must be reviewed separately before refreshing these hashes.
BASELINE = {
    "kernels/aggregate/aggregate.cpp": "059a16c426a07cf047c3075e172a89456d768fca89cb7f055417c9657ef534a9",
    "kernels/combine/combine.cpp": "b80686b6bd4d468657efbc712d38ea3661af98885be6cf0cfac6871465600bd1",
    "kernels/combine/combine_udma.cpp": "e4fc2416a6d17193d185cf98e5c612a7cbd621edaa9e3ad2abdf7841a14aeaf5",
    "kernels/dispatch/dispatch.cpp": "fa8286f2d3833704523b866e7900531b9ea4f3ab6a53dc0df5904d9bff7cef07",
    "kernels/dispatch/dispatch_layout.cpp": "52bb40ff2c8d180c484387b54e69cb7d2d4e4bc9ab822931606a06ea8553e06a",
    # Original UDMA source minus its Host-only v1_token_schedule_abi_3 marker.
    # The AOT support library now owns a distinct marker in kernels/abi.cpp.
    "kernels/dispatch/dispatch_udma.cpp": "cadbbe03fa03dee02cfc71e283ab5f002ecef9709a4076c156409e60c9760d40",
    "kernels/expand/expand.cpp": "16fa778493790a24632ee8f6203682154b94074f67550ab895926d0b8c5aeb77",
    "kernels/notify/notify_combine.cpp": "698398ab9b65e088d7c06643c69761a8a216b8f4d1594c5962576f30f8e70bb4",
    "kernels/notify/notify_dispatch.cpp": "6879c9c205642aa35734b8ae15e8b089b202aa5b0e8a55e36cedc3b05171752e",
}


@pytest.fixture(scope="module")
def compiler():
    value = shutil.which("g++") or shutil.which("clang++")
    if not value:
        pytest.skip("Requires a host C++ compiler")
    return value


@pytest.fixture(scope="module")
def registry(compiler, tmp_path_factory):
    directory = tmp_path_factory.mktemp("direct-jit-spec")
    source = directory / "registry.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <iostream>
#include <set>
#include <string>
#include <jit/spec.hpp>
using namespace ascend_deepep::direct_jit;
int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "source") {
        std::cout << source(static_cast<Entry>(std::stoul(argv[2])));
        return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "transport") {
        for (const auto entry : entries_for_transport(parse_transport(argv[2])))
            std::cout << static_cast<unsigned>(entry) << '\n';
        return 0;
    }
    for (const unsigned id : {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 9U, 10U,
                              11U, 13U, 14U, 15U, 16U, 17U, 18U, 19U, 0xffffffffU}) {
        bool rejected = false;
        try { source(static_cast<Entry>(id)); }
        catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    for (const auto invalid : {"", "mesh_clos", "auto", "MTE", "udma;anything"}) {
        bool rejected = false;
        try { parse_transport(invalid); }
        catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    bool rejected = false;
    try { entries_for_transport(static_cast<Transport>(42)); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    std::set<unsigned> ids;
    std::set<std::string> symbols;
    for (const auto& item : descriptors) {
        const auto id = static_cast<unsigned>(item.entry);
        assert(&descriptor(item.entry) == &item);
        assert(ids.insert(id).second && symbols.insert(item.symbol).second);
        std::cout << id << '\t' << item.source_path << '\t' << item.symbol << '\t'
                  << item.ub_bytes << '\t' << static_cast<unsigned>(item.category) << '\n';
    }
}
""",
        encoding="utf-8",
    )
    binary = directory / ("registry.exe" if sys.platform == "win32" else "registry")
    subprocess.run([compiler, "-std=c++17", "-I" + str(DIRECT), str(source), "-o", str(binary)], check=True, timeout=60)
    records = {}
    for line in subprocess.check_output([str(binary)], text=True, timeout=30).splitlines():
        identity, path, symbol, ub, category = line.split("\t")
        records[int(identity)] = {"path": path, "symbol": symbol, "ub": int(ub), "category": int(category)}
    return binary, records


@pytest.fixture(scope="module")
def preprocess_root(tmp_path_factory):
    directory = tmp_path_factory.mktemp("direct-device-preprocess")
    for source in KERNELS:
        destination = directory / source.relative_to(DIRECT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        for included in re.findall(r'^#include [<"]([^>"]+)[>"]', source.read_text(), re.MULTILINE):
            stub = directory / included
            stub.parent.mkdir(parents=True, exist_ok=True)
            stub.touch()
    return directory


def preprocess(compiler, directory, source):
    return subprocess.check_output(
        [compiler, "-E", "-P", "-x", "c++", "-nostdinc", "-I" + str(directory), "-"],
        input=source,
        text=True,
        timeout=30,
    )


def test_registry_contains_only_payload_dispatch_and_keeps_other_device_entries(registry):
    _, records = registry
    original = [symbol for path in KERNELS for symbol in GLOBAL.findall(path.read_text())]
    assert len(original) == len(set(original)) == 18
    assert set(records) == {8, 12}
    assert {item["symbol"] for item in records.values()} == {"v1_dispatch_kernel", "v1_dispatch_udma_kernel"}
    for item in records.values():
        assert item["symbol"] in GLOBAL.findall((DIRECT / item["path"]).read_text())
        assert item["ub"] == 216 * 1024


@pytest.mark.parametrize("entry", [8, 12])
def test_jit_preprocessing_keeps_one_selected_entry_and_no_aot_wrappers(entry, registry, compiler, preprocess_root):
    binary, records = registry
    generated = subprocess.check_output([str(binary), "source", str(entry)], text=True, timeout=30)
    assert re.findall(r"#include <([^>]+)>", generated) == [records[entry]["path"]]
    compiled = preprocess(compiler, preprocess_root, generated)
    assert GLOBAL.findall(compiled) == [records[entry]["symbol"]]
    assert not re.search(r"\b\w+_kernel_do\s*\(", compiled)
    assert "<<<" not in compiled
    assert "v1_token_schedule_abi_3" not in compiled


@pytest.mark.parametrize("relative", sorted(BASELINE))
def test_aot_preprocessing_retains_all_original_entries(relative, compiler, preprocess_root):
    original = (DIRECT / relative).read_text()
    compiled = preprocess(compiler, preprocess_root, f"#include <{relative}>\n")
    assert GLOBAL.findall(compiled) == GLOBAL.findall(original)
    assert compiled.count("<<<") == original.count("<<<")


@pytest.mark.parametrize("transport,expected_entry", [("mte", 8), ("udma", 12)])
def test_transport_selection_excludes_other_topology_and_non_dispatch(registry, transport, expected_entry):
    binary, records = registry
    entries = [
        int(line)
        for line in subprocess.check_output([str(binary), "transport", transport], text=True, timeout=30).splitlines()
    ]
    assert entries == [expected_entry]
    assert records[expected_entry]["path"] == (
        "kernels/dispatch/dispatch.cpp" if transport == "mte" else "kernels/dispatch/dispatch_udma.cpp"
    )


@pytest.mark.parametrize("relative", sorted(BASELINE))
def test_topology_jit_guards_preserve_original_algorithm_bytes(relative):
    source = (DIRECT / relative).read_text(encoding="utf-8")
    restored = re.sub(
        r"^#if !defined\(DEEPEP_DIRECT_JIT_ENTRY\)[^\n]*\n|^#endif  // DEEPEP_DIRECT_JIT_ENTRY\n?",
        "",
        source,
        flags=re.MULTILINE,
    )
    assert hashlib.sha256(restored.encode()).hexdigest() == BASELINE[relative]
