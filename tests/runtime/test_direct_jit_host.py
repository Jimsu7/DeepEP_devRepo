"""Run the actual Direct Host bridge against fake ACL/compiler interfaces.

This verifies preparation transactions, context separation, ABI source checks,
and every argument passed by the two payload Dispatch C launch interfaces. It does not
establish actual Bisheng compilation, ACL loading, or NPU communication.
"""

from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
DIRECT = ROOT / "csrc/direct"

BACKEND_STUB = r"""
#pragma once
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
using aclrtContext = void*;
using aclrtStream = void*;
namespace fake {
inline aclrtContext context = reinterpret_cast<void*>(1);
inline int arch = 3510;
inline unsigned lookups = 0, builds = 0, loads = 0, launches = 0;
inline std::string fail_build, fail_load, wrong_symbol, last_name;
inline std::map<std::string, std::string> artifacts;
inline std::vector<std::vector<uint8_t>> arguments;
inline void* stream = nullptr;
inline uint32_t blocks = 0, ub = 0;
template<class T> std::vector<uint8_t> bytes(const T& value) {
    std::vector<uint8_t> result(sizeof(value));
    std::memcpy(result.data(), &value, sizeof(value));
    return result;
}
}
inline int aclrtGetCurrentContext(aclrtContext* result) { *result = fake::context; return 0; }
#define DJ_ACL_CHECK(expr) do { if ((expr) != 0) throw std::runtime_error("ACL failure"); } while (0)
#define DJ_HOST_ASSERT(expr, ...) do { if (!(expr)) throw std::runtime_error("Host assertion"); } while (0)
namespace deep_jit {
struct Ascend {};
struct Config {
    Config(std::filesystem::path root, std::string prefix, std::string signature,
           std::vector<std::filesystem::path>, std::vector<std::string>) {
        assert(root.is_absolute() && prefix == "EP_DIRECT" && !signature.empty());
    }
};
namespace ascend {
struct Kernel {
    std::string name;
    static std::vector<std::string> parse_kernel_names(const std::filesystem::path& path) {
        if (!fake::wrong_symbol.empty()) return {fake::wrong_symbol};
        const auto name = path.parent_path().filename().string();
        return {name.substr(std::string("direct_").size())};
    }
};
}
struct Options { std::vector<std::string> bisheng_flags; };
struct LaunchOptions {
    void* stream = nullptr;
    unsigned num_blocks = 0;
    unsigned num_ubuf_bytes = 0;
    unsigned num_launch_timeout_secs = 0;
};
template<class Backend> struct Runtime {
    struct Device { int get_npu_arch() { return fake::arch; } } device;
    Options default_compiler_options;
    LaunchOptions default_launch_options;
    explicit Runtime(Config) {}
    std::filesystem::path compile_without_load(const std::string& name, const std::string& source) {
        ++fake::lookups;
        if (fake::fail_build == name) throw std::runtime_error("compile failed");
        if (!fake::artifacts.contains(name)) {
            ++fake::builds;
            fake::artifacts.emplace(name, source);
        }
        assert(fake::artifacts.at(name) == source);
        assert(source.find("static_assert(__builtin_offsetof") != std::string::npos);
        return std::filesystem::path("/fake") / name;
    }
    std::shared_ptr<ascend::Kernel> compile(const std::string& name, const std::string& source) {
        assert(fake::artifacts.at(name) == source);
        if (fake::fail_load == name) throw std::runtime_error("load failed");
        ++fake::loads;
        return std::make_shared<ascend::Kernel>(ascend::Kernel{name});
    }
    template<class... Args> void launch(const std::shared_ptr<ascend::Kernel>& kernel,
                                       const LaunchOptions& options, const Args&... args) {
        ++fake::launches;
        fake::last_name = kernel->name;
        fake::arguments = {fake::bytes(args)...};
        fake::stream = options.stream;
        fake::blocks = options.num_blocks;
        fake::ub = options.num_ubuf_bytes;
    }
};
}
"""

HARNESS = r"""
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <deep_jit/backend/ascend/backend.hpp>
#include <common/launch.h>
#include <common/types.h>
#include <jit/abi.hpp>
#include <jit/runtime.hpp>
using namespace ascend_deepep::direct_jit;
extern "C" uint64_t direct_aot_dispatch_split_abi_1();
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}
void wrapper_cases();
void launch_mte();
void reject_unready_mte_launch();
void reject_unready_udma_launch();
void reject_invalid_launches();
int main(int argc, char** argv) {
    assert(argc == 2);
    assert(direct_aot_dispatch_split_abi_1() ==
           ((uint64_t(sizeof(ascend_deepep::DispatchTilingData)) << 32) |
            uint64_t(sizeof(ascend_deepep::CombineTilingData))));
    const std::string mode = argv[1];
    if (mode == "abi") { std::cout << abi_source(); return 0; }
    rejects([] { ensure_prepared("mte"); });
    reject_unready_mte_launch();
    initialize("/package", {"/package/jit/direct"}, "complete-source-sdk-signature");
    rejects([] { initialize("/package", {}, "another-signature"); });
    rejects([] { prepare("mesh_clos"); });
    assert(fake::builds == 0 && fake::launches == 0);
    if (mode == "cache") {
        prepare("mte");
        assert(fake::builds == 1 && fake::loads == 1);
        assert(prepared_entries().size() == 1);
        ensure_prepared("mte");
        rejects([] { ensure_prepared("udma"); });
        for (int i = 0; i != 100; ++i) prepare("mte");
        assert(fake::lookups == 1 && fake::loads == 1);
        for (const auto& [name, code] : fake::artifacts)
            assert(name.find("udma") == std::string::npos);
        prepare("udma");
        assert(fake::builds == 2 && fake::loads == 2);
        assert(prepared_entries().size() == 2);
    } else if (mode == "compile-failure" || mode == "load-failure" || mode == "wrong-symbol") {
        const std::string failing = "direct_v1_dispatch_kernel";
        if (mode == "compile-failure") fake::fail_build = failing;
        if (mode == "load-failure") fake::fail_load = failing;
        if (mode == "wrong-symbol") fake::wrong_symbol = "unrelated_kernel";
        rejects([] { prepare("mte"); });
        rejects([] { ensure_prepared("mte"); });
        assert(prepared_entries().empty());
        reject_unready_mte_launch();
        assert(fake::launches == 0);
        fake::fail_build.clear(); fake::fail_load.clear(); fake::wrong_symbol.clear();
        prepare("mte");
        ensure_prepared("mte");
        assert(fake::builds == 1 && fake::loads == 1);
    } else if (mode == "other-failure") {
        prepare("mte");
        fake::fail_build = "direct_v1_dispatch_udma_kernel";
        rejects([] { prepare("udma"); });
        ensure_prepared("mte");
        rejects([] { ensure_prepared("udma"); });
        assert(prepared_entries().size() == 1);
        reject_unready_udma_launch();
        assert(fake::launches == 0);
        launch_mte();
        assert(fake::launches == 1);  // The already-ready MTE backend remains usable.
        fake::fail_build.clear();
        prepare("udma");
        assert(prepared_entries().size() == 2 && fake::builds == 2);
    } else if (mode == "context") {
        prepare("udma");
        assert(fake::builds == 1 && fake::loads == 1);
        fake::context = reinterpret_cast<void*>(2);
        rejects([] { ensure_prepared("udma"); });
        reject_unready_udma_launch();
        prepare("udma");
        assert(fake::builds == 1 && fake::loads == 2);
        fake::context = reinterpret_cast<void*>(1);
        ensure_prepared("udma");
        prepare("udma");
        assert(fake::loads == 2);
    } else if (mode == "architecture") {
        fake::arch = 2201;
        rejects([] { prepare("mte"); });
        assert(fake::builds == 0);
        fake::arch = 3510;
        fake::context = nullptr;
        rejects([] { prepare("mte"); });
        assert(fake::builds == 0);
    } else if (mode == "wrappers") {
        prepare("mte"); prepare("udma");
        wrapper_cases();
        assert(fake::launches == 2);
        const auto builds = fake::builds, lookups = fake::lookups, loads = fake::loads;
        wrapper_cases();
        assert(fake::launches == 4);
        assert(fake::builds == builds && fake::lookups == lookups && fake::loads == loads);
        reject_invalid_launches();
        assert(fake::launches == 4);
    } else {
        assert(false);
    }
}
"""


def _wrapper_cases():
    """Use the retained original AOT bodies as the argument-order oracle."""
    header = (DIRECT / "common/launch.h").read_text()
    declarations = [
        (name, parameters)
        for name, parameters in re.findall(r"ASCEND_DEEPEP_EXPORT void (\w+)\((.*?)\);", header, re.S)
        if name in {"v1_dispatch_kernel_do", "v1_dispatch_udma_plan_kernel_do"}
    ]
    device_sources = "\n".join(p.read_text() for p in (DIRECT / "kernels").rglob("*.cpp"))
    lines = ["void wrapper_cases() {\n"]
    helpers = []
    invalid = []
    assert len(declarations) == 2
    for index, (name, parameters) in enumerate(declarations):
        body = re.search(
            r'extern "C" ASCEND_DEEPEP_EXPORT void ' + name + r"\(.*?\)\s*\{(.*?)\n\}",
            device_sources,
            re.S,
        ).group(1)
        device_call = re.search(r"(\w+)<<<.*?>>>\((.*?)\);", body, re.S)
        symbol, original_args = device_call.groups()
        params = [re.search(r"(\w+)$", p.strip()).group(1) for p in parameters.split(",")]
        tiling = re.search(r"reinterpret_cast<ascend_deepep::(\w+)\*>\(tiling_host\)", body)
        lines.append("{\n")
        if tiling:
            lines.append(
                f"ascend_deepep::{tiling.group(1)} td; "
                f"std::fill_n(reinterpret_cast<unsigned char*>(&td), sizeof(td), {index + 1});\n"
            )
        values = {
            "block_dim": "uint32_t(13)",
            "stream": "reinterpret_cast<void*>(0x4321)",
            "ffts_addr": "uint64_t(0xfedcba9876543210ULL)",
        }
        for number, param in enumerate(params):
            if param not in values:
                values[param] = (
                    "reinterpret_cast<uint8_t*>(&td)"
                    if param == "tiling_host"
                    else f"reinterpret_cast<uint8_t*>({0x1000 + number * 32})"
                )
        call = ", ".join(values[param] for param in params)
        transport = "udma" if "udma" in name else "mte"
        helpers.append(
            f"void reject_unready_{transport}_launch() {{\n"
            f"ascend_deepep::{tiling.group(1)} td{{}};\n"
            f"rejects([&] {{ {name}({call}); }});\n}}\n"
        )
        if transport == "mte":
            helpers.append(f"void launch_mte() {{\nascend_deepep::{tiling.group(1)} td{{}};\n{name}({call});\n}}\n")
        invalid.append("{\n" + f"ascend_deepep::{tiling.group(1)} td{{}};\n")
        for field, value in (("block_dim", "uint32_t(0)"), ("tiling_host", "nullptr")):
            args = ", ".join(value if param == field else values[param] for param in params)
            invalid.append(f"rejects([&] {{ {name}({args}); }});\n")
        invalid.append("}\n")
        lines.append(f"{name}({call});\n")
        expected = []
        for arg in original_args.split(","):
            arg = arg.strip()
            expected.append("fake::bytes(td)" if arg == "*td" else f"fake::bytes({values[arg]})")
        lines.append("assert(fake::arguments == std::vector<std::vector<uint8_t>>({" + ", ".join(expected) + "}));\n")
        lines.append(f'assert(fake::last_name == "direct_{symbol}");\n')
        ub = 216 * 1024
        lines.append(
            f"assert(fake::blocks == 13 && fake::ub == {ub} && fake::stream == reinterpret_cast<void*>(0x4321));\n"
        )
        lines.append("}\n")
    return "".join(lines) + "}\n" + "".join(helpers) + "void reject_invalid_launches() {\n" + "".join(invalid) + "}\n"


@pytest.fixture(scope="module")
def host_bridge(tmp_path_factory):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("Requires a host C++ compiler")
    directory = tmp_path_factory.mktemp("direct-jit-host")
    stub = directory / "include/deep_jit/backend/ascend/backend.hpp"
    stub.parent.mkdir(parents=True)
    stub.write_text(BACKEND_STUB)
    harness = directory / "harness.cpp"
    harness.write_text(HARNESS + _wrapper_cases())
    binary = directory / "host-bridge"
    flags = [
        compiler,
        "-std=c++20",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-DASCEND_DEEPEP_ARCH=3510",
        "-DEP_NUM_TOPK_IDX_BITS=64",
    ]
    subprocess.run(
        [
            *flags,
            "-I" + str(directory / "include"),
            "-I" + str(DIRECT),
            str(DIRECT / "jit/runtime.cpp"),
            str(DIRECT / "kernels/abi.cpp"),
            str(harness),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary, flags, directory


@pytest.mark.parametrize(
    "case",
    [
        "cache",
        "compile-failure",
        "load-failure",
        "wrong-symbol",
        "other-failure",
        "context",
        "architecture",
        "wrappers",
    ],
)
def test_actual_host_bridge_with_fake_compiler_and_acl(host_bridge, case):
    binary, _, _ = host_bridge
    subprocess.run([str(binary), case], check=True, capture_output=True, text=True)


def test_generated_device_checks_cover_dispatch_tiling_and_reject_abi_mismatch(host_bridge):
    binary, flags, directory = host_bridge
    generated = subprocess.check_output([str(binary), "abi"], text=True)
    types = (DIRECT / "common/types.h").read_text()
    expected = set()
    for name, body in re.findall(r"struct alignas\(8\) (\w+) \{(.*?)\n\};", types, re.S):
        if name != "DispatchTilingData":
            assert f"sizeof(ascend_deepep::{name})" not in generated
            assert f"alignof(ascend_deepep::{name})" not in generated
            continue
        assert f"sizeof(ascend_deepep::{name})" in generated
        assert f"alignof(ascend_deepep::{name})" in generated
        expected.update((name, field) for field in re.findall(r"uint(?:32|64)_t\s+(\w+)", body))
    actual = set(re.findall(r"__builtin_offsetof\(ascend_deepep::(\w+), (\w+)\)", generated))
    assert actual == expected
    source = directory / "device-abi.cpp"
    source.write_text(generated + "\nint main() {}\n")
    command = [*flags, "-I" + str(DIRECT), "-fsyntax-only", str(source)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    changed = generated.replace("DispatchTilingData, rank) == 0", "DispatchTilingData, rank) == 4")
    assert changed != generated
    source.write_text(changed)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and "static assertion failed" in result.stderr
