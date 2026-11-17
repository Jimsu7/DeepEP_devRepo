#include "jit/runtime.hpp"

#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <deep_jit/backend/ascend/backend.hpp>

#include "common/launch.h"
#include "common/types.h"
#include "jit/abi.hpp"
#include "jit/spec.hpp"

namespace ascend_deepep::direct_jit {
namespace {

using Runtime = deep_jit::Runtime<deep_jit::Ascend>;
using Kernel = deep_jit::ascend::Kernel;
std::mutex mutex;
std::unique_ptr<deep_jit::Config> configuration;

struct ContextState {
    std::shared_ptr<Runtime> runtime;
    std::map<Entry, std::shared_ptr<Kernel>> kernels;
    std::set<Transport> ready;

    explicit ContextState(const deep_jit::Config& config) : runtime(std::make_shared<Runtime>(config))
    {
        DJ_HOST_ASSERT(runtime->device.get_npu_arch() == 3510, "Direct JIT requires dav-3510");
        runtime->default_compiler_options.bisheng_flags = std::vector<std::string>{"--cce-aicore-only",
                                                                                   "--cce-auto-infer-kernel-type=false",
                                                                                   "-DASCEND_DEEPEP_ARCH=3510",
                                                                                   "-DEP_NUM_TOPK_IDX_BITS=64",
                                                                                   "-DUSE_SIMT",
                                                                                   "-DL2_CACHE_HINT"};
        runtime->default_launch_options.num_launch_timeout_secs = 0;
    }
};

// ACL contexts must remain alive while these handles are cached. In-process
// source/SDK reload and destroyed/recreated contexts are not supported.
std::map<aclrtContext, std::unique_ptr<ContextState>> contexts;

ContextState& current_state(bool create)
{
    if (!configuration) throw std::runtime_error("Direct JIT must be initialized before SHMEM construction");
    aclrtContext context = nullptr;
    DJ_ACL_CHECK(aclrtGetCurrentContext(&context));
    DJ_HOST_ASSERT(context != nullptr, "Direct JIT requires an active ACL context");
    auto found = contexts.find(context);
    if (found == contexts.end()) {
        if (!create) throw std::runtime_error("Direct JIT is not prepared in the current ACL context");
        auto state = std::make_unique<ContextState>(*configuration);
        found = contexts.emplace(context, std::move(state)).first;
    }
    return *found->second;
}

bool entry_ready(const ContextState& state, Entry entry)
{
    const auto category = descriptor(entry).category;
    return state.ready.contains(category == Category::Udma ? Transport::Udma : Transport::Mte);
}

template <typename... Args>
void launch(Entry entry, uint32_t block_dim, void* stream, const Args&... args)
{
    if (block_dim == 0) throw std::invalid_argument("Direct JIT block_dim must be positive");
    std::shared_ptr<Runtime> runtime;
    std::shared_ptr<Kernel> kernel;
    {
        std::lock_guard lock(mutex);
        auto& state = current_state(false);
        const auto found = state.kernels.find(entry);
        if (!entry_ready(state, entry) || found == state.kernels.end())
            throw std::runtime_error("Direct JIT kernel was not prepared before SHMEM construction");
        runtime = state.runtime;
        kernel = found->second;
    }
    // This path deliberately never compiles: one rank must not enter a
    // communication kernel while another rank is still preparing or failing.
    runtime->launch(kernel,
                    {.stream = static_cast<aclrtStream>(stream),
                     .num_blocks = block_dim,
                     .num_ubuf_bytes = descriptor(entry).ub_bytes},
                    args...);
}

template <typename T>
T tiling_value(const uint8_t* address)
{
    static_assert(std::is_trivially_copyable_v<T>);
    if (address == nullptr) throw std::invalid_argument("Direct JIT tiling_host must not be null");
    T value;
    std::memcpy(&value, address, sizeof(value));
    return value;
}

}  // namespace

void initialize(const std::string& library_root, const std::vector<std::string>& include_dirs,
                const std::string& signature)
{
    std::lock_guard lock(mutex);
    if (configuration) throw std::runtime_error("Direct JIT is already configured");
    if (signature.empty()) throw std::invalid_argument("Direct JIT dependency signature must not be empty");
    std::vector<std::filesystem::path> paths;
    for (const auto& path : include_dirs) paths.emplace_back(path);
    configuration =
        std::make_unique<deep_jit::Config>(library_root, "EP_DIRECT", signature, paths, std::vector<std::string>{});
}

void prepare(const std::string& transport)
{
    const auto selected = parse_transport(transport);
    std::lock_guard lock(mutex);
    auto& state = current_state(true);
    if (state.ready.contains(selected)) return;
    for (const auto entry : entries_for_transport(selected)) {
        if (state.kernels.contains(entry)) continue;
        const auto& item = descriptor(entry);
        const auto name = std::string("direct_") + item.symbol;
        const auto code = abi_source() + source(entry);
        const auto artifact = state.runtime->compile_without_load(name, code);
        const auto symbols = Kernel::parse_kernel_names(artifact / "kernel.o");
        if (symbols.size() != 1 || symbols.front() != item.symbol)
            throw std::runtime_error("Direct JIT artifact has an unexpected device entry");
        // The artifact has been committed, so compile() now loads it from the
        // disk cache; it does not invoke Bisheng again.
        auto kernel = state.runtime->compile(name, code);
        state.kernels.emplace(entry, std::move(kernel));
    }
    // Publish readiness only after the selected payload Dispatch was loaded.
    state.ready.insert(selected);
}

void ensure_prepared(const std::string& transport)
{
    const auto selected = parse_transport(transport);
    std::lock_guard lock(mutex);
    if (!current_state(false).ready.contains(selected))
        throw std::runtime_error("Prepare Direct JIT collectively before constructing a SHMEM Buffer");
}

std::vector<std::string> prepared_entries()
{
    std::lock_guard lock(mutex);
    auto& state = current_state(false);
    std::vector<std::string> result;
    for (const auto& [entry, kernel] : state.kernels) {
        (void)kernel;
        if (entry_ready(state, entry)) result.emplace_back(descriptor(entry).symbol);
    }
    return result;
}

}  // namespace ascend_deepep::direct_jit

using ascend_deepep::direct_jit::Entry;
using ascend_deepep::direct_jit::launch;
using ascend_deepep::direct_jit::tiling_value;

// The C ABI below is the existing Host launcher contract. Only submission
// changes; each tiling record is copied into the ACL argument buffer BY VALUE.

extern "C" ASCEND_DEEPEP_EXPORT void v1_dispatch_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* x, uint8_t* x_scales, uint8_t* topk_idx, uint8_t* topk_weights,
    uint8_t* num_tokens_per_rank, uint8_t* is_token_in_rank, uint8_t* num_tokens_per_expert, uint8_t* symmetric_buffer,
    uint8_t* recv_x, uint8_t* recv_x_scales, uint8_t* recv_topk_idx, uint8_t* recv_topk_weights,
    uint8_t* rank_prefix_matrix, uint8_t* channel_prefix_matrix, uint8_t* recv_channel_prefix_matrix,
    uint8_t* recv_src_idx, uint8_t* send_head, uint8_t* tiling_host, uint64_t ffts_addr)
{
    launch(Entry::DispatchMte, block_dim, stream, x, x_scales, topk_idx, topk_weights, num_tokens_per_rank,
           is_token_in_rank, num_tokens_per_expert, symmetric_buffer, recv_x, recv_x_scales, recv_topk_idx,
           recv_topk_weights, rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx,
           send_head, tiling_value<ascend_deepep::DispatchTilingData>(tiling_host), ffts_addr);
}

extern "C" ASCEND_DEEPEP_EXPORT void v1_dispatch_udma_plan_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* x, uint8_t* x_scales, uint8_t* topk_idx, uint8_t* topk_weights,
    uint8_t* num_tokens_per_rank, uint8_t* is_token_in_rank, uint8_t* num_tokens_per_expert, uint8_t* symmetric_buffer,
    uint8_t* recv_x, uint8_t* recv_x_scales, uint8_t* recv_topk_idx, uint8_t* recv_topk_weights,
    uint8_t* rank_prefix_matrix, uint8_t* channel_prefix_matrix, uint8_t* recv_channel_prefix_matrix,
    uint8_t* recv_src_idx, uint8_t* send_head, uint8_t* dispatch_plan, uint8_t* tiling_host, uint64_t ffts_addr)
{
    launch(Entry::DispatchUdma, block_dim, stream, x, x_scales, topk_idx, topk_weights, num_tokens_per_rank,
           is_token_in_rank, num_tokens_per_expert, symmetric_buffer, recv_x, recv_x_scales, recv_topk_idx,
           recv_topk_weights, rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx,
           send_head, dispatch_plan, tiling_value<ascend_deepep::DispatchTilingData>(tiling_host), ffts_addr);
}
