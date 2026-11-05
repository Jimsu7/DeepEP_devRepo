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
    prepare = launch_text[
        launch_text.index("inline Prepared prepare(") : launch_text.index("inline DispatchSpec spec_for(")
    ]
    state_text = (ROOT / "csrc/runtime/jit.hpp").read_text(encoding="utf-8")
    cache_field = next(line for line in state_text.splitlines() if "prepared_dispatch;" in line)
    source = tmp_path / "prepared_cache.cpp"
    source.write_text(
        r"""
#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
using aclrtContext = void*;
namespace deep_jit::ascend { struct Kernel {}; }
struct Runtime {
    using Kernel = deep_jit::ascend::Kernel;
    unsigned compile_calls = 0;
    bool fail = false;
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
struct Prepared {
    std::shared_ptr<Runtime> runtime;
    std::shared_ptr<deep_jit::ascend::Kernel> kernel;
    DispatchSpec spec{};
    aclrtContext context = nullptr;
};
"""
        + prepare
        + r"""
int main() {
    const DispatchSpec baseline{2,128,2,false,true,false};
    const auto original = prepare(baseline);
    assert(original.context == jit::first.context);
    assert(jit::first.runtime->compile_calls == 1 && source_calls == 1);
    for (unsigned i = 0; i < 100; ++i) {
        const auto hit = prepare(baseline);
        assert(hit.kernel == original.kernel && hit.runtime == original.runtime);
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

    jit::current = &jit::second;
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
    subprocess.run([compiler, "-std=c++20", str(source), "-o", str(binary)], check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=30)
