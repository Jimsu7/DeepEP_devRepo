// Copyright (c) 2026, Lu Lu
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <deep_jit/backend/ascend/backend.hpp>

namespace deepep::jit {
using Runtime = deep_jit::Runtime<deep_jit::Ascend>;
inline std::mutex mutex;
inline std::unique_ptr<deep_jit::Config> configuration;

struct ContextState {
    aclrtContext context;
    std::shared_ptr<Runtime> runtime;
    // Configuration and compiler options are immutable after construction.
    // Keep loaded Dispatch handles under their full template identity, so warm
    // calls do not regenerate source or hash it before DeepJIT's memory lookup.
    std::map<std::array<uint32_t, 6>, std::shared_ptr<Runtime::Kernel>> prepared_dispatch;

    explicit ContextState(aclrtContext current, const deep_jit::Config& config)
        : context(current), runtime(std::make_shared<Runtime>(config))
    {
        DJ_HOST_ASSERT(runtime->device.get_npu_arch() == 3510, "Dispatch JIT requires dav-3510");
        runtime->default_compiler_options.bisheng_flags =
            std::vector<std::string>{"--cce-aicore-only", "--cce-auto-infer-kernel-type=false", "-DCATLASS_ARCH=3510",
                                     "-DUSE_SIMT", "-DL2_CACHE_HINT"};
        // Retain the original dispatch launch; Session transport timeouts remain active.
        runtime->default_launch_options.num_launch_timeout_secs = 0;
    }
};
// Loaded kernel handles belong to their ACL context. Contexts must remain alive
// until process exit; destroying/recreating contexts in the same process is unsupported.
inline std::unordered_map<aclrtContext, std::unique_ptr<ContextState>> contexts;

inline void initialize(const std::string& library_root, const std::vector<std::string>& include_dirs,
                       const std::string& signature)
{
    std::lock_guard lock(mutex);
    if (configuration) throw std::runtime_error("DeepEP JIT is already configured");
    std::vector<std::filesystem::path> paths;
    for (const auto& path : include_dirs) paths.emplace_back(path);
    // Original device headers contain quoted relative includes. Python hashes
    // the complete installed source/SDK tree; Bisheng resolves those includes.
    configuration =
        std::make_unique<deep_jit::Config>(library_root, "EP", signature, paths, std::vector<std::string>{});
}

// The caller holds mutex: DeepJIT parser/memory-cache accesses are serialized.
inline ContextState& current_state()
{
    if (!configuration) throw std::runtime_error("DeepEP JIT must be initialized through deep_ep");
    aclrtContext context = nullptr;
    DJ_ACL_CHECK(aclrtGetCurrentContext(&context));
    DJ_HOST_ASSERT(context != nullptr, "Dispatch JIT requires an active ACL context");
    auto& state = contexts[context];
    if (!state) state = std::make_unique<ContextState>(context, *configuration);
    return *state;
}
}  // namespace deepep::jit
