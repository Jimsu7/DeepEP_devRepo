"""Exercise production native preparation with fake context/compiler backends.

This proves cache isolation, specialization identity and failure retry semantics;
it does not establish ACL loading or device launch correctness.
"""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_native_prepared_cache_isolates_contexts_and_retries_failed_loads(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("Requires a host C++ compiler")
    spec_text = (ROOT / "csrc/kernels/dispatch/jit_spec.hpp").read_text(encoding="utf-8")
    spec = spec_text[spec_text.index("struct DispatchSpec {") : spec_text.index("inline const char* tag(")]
    launch_text = (ROOT / "csrc/kernels/dispatch/jit_launcher.hpp").read_text(encoding="utf-8")
    prepared = launch_text[launch_text.index("struct Prepared {") : launch_text.index("inline Prepared prepare(")]
    prepare = launch_text[
        launch_text.index("inline Prepared prepare(") : launch_text.index("inline DispatchSpec spec_for(")
    ]
    launch = launch_text[launch_text.index("inline DispatchSpec spec_for(") : launch_text.index("struct Metrics {")]
    state_text = (ROOT / "csrc/runtime/jit.hpp").read_text(encoding="utf-8")
    cache_field = next(line for line in state_text.splitlines() if "prepared_dispatch;" in line)
    source = tmp_path / "prepared_cache.cpp"
    source.write_text(
        r"""
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <type_traits>
#include <cstdio>
#include "kernels/dispatch/core_policy.hpp"
using deepep::dispatch_jit::select_block_count;
using aclrtContext = void*;
using aclrtStream = void*;
namespace ascend_deepep {
struct DispatchTiling { uint32_t world, hidden, topk, fp8; uint64_t profile; };
}
struct LaunchConfig { aclrtStream stream; uint32_t num_blocks, num_ubuf_bytes; };
#define DJ_ACL_CHECK(expression) do { if ((expression) != 0) throw std::runtime_error("ACL failure"); } while(0)
#define DJ_HOST_ASSERT(expression, message) do { if (!(expression)) throw std::runtime_error(message); } while(0)
uint32_t available_aivs = 64;
uint32_t QueryAvailableVectorCores() { return available_aivs; }
namespace deep_jit::ascend { struct Kernel {}; }
struct Runtime {
    using Kernel = deep_jit::ascend::Kernel;
    unsigned compile_calls = 0, launch_calls = 0, launched_blocks = 0;
    bool fail = false;
    template<class... Args> void launch(std::shared_ptr<Kernel>, LaunchConfig config, Args...) {
        ++launch_calls;
        launched_blocks = config.num_blocks;
        assert(config.num_ubuf_bytes == 192 * 1024);
    }
    std::shared_ptr<Kernel> compile(const char*, const std::string&) {
        ++compile_calls;
        if (fail) throw std::runtime_error("load failed");
        return std::make_shared<Kernel>();
    }
};
"""
        + spec
        + r"""
unsigned source_calls = 0;
const char* tag(const DispatchSpec&) { return "dispatch"; }
std::string source(const DispatchSpec&) { ++source_calls; return "kernel"; }
namespace jit {
std::mutex mutex;
struct State {
    aclrtContext context;
    std::shared_ptr<Runtime> runtime;
"""
        + cache_field
        + r"""
};
State first{reinterpret_cast<void*>(1), std::make_shared<Runtime>()};
State second{reinterpret_cast<void*>(2), std::make_shared<Runtime>()};
State* current = &first;
State& current_state() { return *current; }
}
int aclrtGetCurrentContext(aclrtContext* context) { *context = jit::current->context; return 0; }
"""
        + prepared
        + prepare
        + launch
        + r"""
int main() {
    const DispatchSpec baseline{2,128,2,false,true,false};
    const auto original = prepare(baseline);
    assert(original.context == jit::first.context);
    assert(original.num_blocks == 64);
    assert(original.spec.identity().size() == 6);
    assert(jit::first.runtime->compile_calls == 1 && source_calls == 1);
    for (unsigned i = 0; i < 100; ++i) {
        const auto hit = prepare(baseline);
        assert(hit.kernel == original.kernel && hit.runtime == original.runtime);
        assert(hit.num_blocks == 64);
    }
    assert(jit::first.runtime->compile_calls == 1 && source_calls == 1);
    for (const DispatchSpec changed : {
             DispatchSpec{4,128,2,false,true,false},
             DispatchSpec{2,256,2,false,true,false},
             DispatchSpec{2,128,3,false,true,false},
             DispatchSpec{2,128,2,true,true,false},
             DispatchSpec{2,128,2,false,false,false},
             DispatchSpec{2,128,2,false,true,true}}) {
        const auto variant = prepare(changed);
        assert(variant.kernel != original.kernel);
        assert(prepare(changed).kernel == variant.kernel);
    }
    assert(jit::first.prepared_dispatch.size() == 7);
    assert(jit::first.runtime->compile_calls == 7 && source_calls == 7);
    assert(prepare(baseline).kernel == original.kernel); // A -> B -> A.
    auto invalid = baseline;
    invalid.world = 3;
    bool rejected = false;
    try { prepare(invalid); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected && jit::first.runtime->compile_calls == 7);

    // Hardware admission precedes both cache hits and cache misses.
    for (uint32_t available : {0U, 31U, 32U, 55U, 56U, 63U}) {
        available_aivs = available;
        for (const DispatchSpec unsupported : {
                 baseline, DispatchSpec{8,128,2,false,true,false},
                 DispatchSpec{2,128,2,false,true,true}}) {
            rejected = false;
            try { prepare(unsupported); } catch (const std::invalid_argument&) { rejected = true; }
            assert(rejected && jit::first.prepared_dispatch.size() == 7);
            assert(jit::first.runtime->compile_calls == 7 && source_calls == 7);
        }
    }
    for (uint32_t available : {64U, 72U, 128U}) {
        available_aivs = available;
        const auto restored = prepare(baseline);
        assert(restored.kernel == original.kernel && restored.num_blocks == 64);
    }
    assert(jit::first.runtime->compile_calls == 7 && source_calls == 7);
    ascend_deepep::DispatchTiling tiling{2,128,2,0,0};
    auto invoke = [&](const Prepared& handle) {
        launch(handle, nullptr, nullptr, nullptr, nullptr, nullptr,
               reinterpret_cast<uint8_t*>(1), nullptr, tiling);
    };
    invoke(original);
    assert(jit::first.runtime->launch_calls == 1 && jit::first.runtime->launched_blocks == 64);
    // A previously prepared object may not bypass the submission-time guard.
    available_aivs = 56;
    rejected = false;
    try { invoke(original); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected && jit::first.runtime->launch_calls == 1);
    available_aivs = 64;
    auto forged = original;
    forged.num_blocks = 56;
    rejected = false;
    try { invoke(forged); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected && jit::first.runtime->launch_calls == 1);

    jit::current = &jit::second;
    rejected = false;
    try { invoke(original); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected && jit::first.runtime->launch_calls == 1);
    jit::second.runtime->fail = true;
    bool failed = false;
    try { prepare(baseline); } catch (const std::runtime_error&) { failed = true; }
    assert(failed && jit::second.prepared_dispatch.empty());
    jit::second.runtime->fail = false;
    const auto other = prepare(baseline);
    assert(other.context == jit::second.context && other.kernel != original.kernel);
    assert(jit::second.runtime->compile_calls == 2);
    assert(prepare(baseline).kernel == other.kernel);
    jit::current = &jit::first;
    assert(prepare(baseline).kernel == original.kernel);
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / ("prepared_cache.exe" if sys.platform == "win32" else "prepared_cache")
    subprocess.run(
        [compiler, "-std=c++20", "-I", str(ROOT / "csrc"), str(source), "-o", str(binary)], check=True, timeout=60
    )
    subprocess.run([str(binary)], check=True, timeout=30)
