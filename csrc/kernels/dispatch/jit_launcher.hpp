// Copyright (c) 2026, Lu Lu
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include "kernels/dispatch/jit_spec.hpp"
#include "kernels/dispatch/tiling.hpp"
#include "runtime/jit.hpp"

namespace deepep::dispatch_jit {
using Runtime = jit::Runtime;
using Clock = std::chrono::steady_clock;

struct Prepared {
    std::shared_ptr<Runtime> runtime;
    std::shared_ptr<deep_jit::ascend::Kernel> kernel;
    DispatchSpec spec{};
    aclrtContext context = nullptr;
};
inline Prepared prepare(const DispatchSpec& spec)
{
    spec.validate();
    std::lock_guard lock(jit::mutex);
    auto& state = jit::current_state();
    const auto identity = spec.identity();
    if (const auto found = state.prepared_dispatch.find(identity); found != state.prepared_dispatch.end())
        return {state.runtime, found->second, spec, state.context};
    // A miss still uses DeepJIT's dependency/option-derived artifact identity.
    // Publish only after compile/load succeeds: failures remain retryable and
    // cannot leave a placeholder that a later preflight could accept.
    auto kernel = state.runtime->compile(tag(spec), source(spec));
    state.prepared_dispatch.emplace(identity, kernel);
    return {state.runtime, std::move(kernel), spec, state.context};
}
inline DispatchSpec spec_for(const ascend_deepep::DispatchTiling& tiling, bool with_weights)
{
    DispatchSpec spec{tiling.world, tiling.hidden, tiling.topk, tiling.fp8 != 0, with_weights, tiling.profile != 0};
    spec.validate();
    return spec;
}
inline void launch(const Prepared& prepared, aclrtStream stream, uint8_t* workspace, uint8_t* destinations,
                   uint8_t* forward, uint8_t* counts, uint8_t* weights, uint8_t* scales,
                   const ascend_deepep::DispatchTiling& tiling)
{
    static_assert(std::is_trivially_copyable_v<ascend_deepep::DispatchTiling>);
    if (prepared.spec != spec_for(tiling, weights != nullptr))
        throw std::invalid_argument("Dispatch specialization differs from runtime tiling or weight presence");
    if ((scales != nullptr) != prepared.spec.fp8)
        throw std::invalid_argument("Dispatch scales differ from the FP8 specialization");
    aclrtContext context = nullptr;
    DJ_ACL_CHECK(aclrtGetCurrentContext(&context));
    DJ_HOST_ASSERT(context == prepared.context, "Dispatch JIT handle belongs to a different ACL context");
    prepared.runtime->launch(prepared.kernel, {.stream = stream, .num_blocks = 64, .num_ubuf_bytes = 192 * 1024},
                             workspace, destinations, forward, counts, weights, scales, tiling);
}

struct Metrics {
    double init_ms = 0, key_ms = 0, build_or_disk_lookup_ms = 0, compile_and_load_ms = 0, memory_compile_hit_us = 0;
    bool disk_hit_before = false, handle_reused = false;
    std::size_t context_kernel_count = 0;
    std::string artifact, key;
};
inline double elapsed_ms(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
inline Metrics diagnostics(const DispatchSpec& spec)
{
    spec.validate();
    std::lock_guard lock(jit::mutex);
    auto start = Clock::now();
    auto& state = jit::current_state();
    Metrics result;
    result.init_ms = elapsed_ms(start);
    auto& runtime = *state.runtime;
    const auto code = source(spec);
    start = Clock::now();
    result.key = runtime.cache_key(code, runtime.default_compiler_options);
    result.key_ms = elapsed_ms(start);
    for (const auto& root : runtime.disk_cache.paths)
        result.disk_hit_before |=
            std::filesystem::exists(root / "cache" / (std::string(tag(spec)) + "." + result.key) / ".committed");
    start = Clock::now();
    result.artifact = runtime.compile_without_load(tag(spec), code).string();
    result.build_or_disk_lookup_ms = elapsed_ms(start);
    result.handle_reused = runtime.mem_cache.cache.contains(result.key);
    start = Clock::now();
    const auto kernel = runtime.compile(tag(spec), code);
    result.compile_and_load_ms = elapsed_ms(start);
    start = Clock::now();
    constexpr int repeats = 100;
    for (int i = 0; i < repeats; ++i)
        DJ_HOST_ASSERT(runtime.compile(tag(spec), code) == kernel, "Memory cache did not reuse the specialized kernel");
    result.memory_compile_hit_us = elapsed_ms(start) * 1000 / repeats;
    result.context_kernel_count = runtime.mem_cache.cache.size();
    return result;
}
}  // namespace deepep::dispatch_jit
