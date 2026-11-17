#pragma once

#include <cstddef>
#include <string>
#include <type_traits>

#include "common/types.h"

namespace ascend_deepep::direct_jit {

// These assertions use the Host compiler's actual layout and are compiled by
// Bisheng with the installed Device sources. Python also verifies that the
// installed types header matches the SHA256 embedded in the native extension.
inline std::string abi_source()
{
    std::string result = "#include <common/types.h>\n";
#define DEEPEP_DIRECT_ABI_TYPE(type)                                                                        \
    static_assert(std::is_standard_layout_v<type> && std::is_trivially_copyable_v<type>);                   \
    result += "static_assert(sizeof(ascend_deepep::" #type ") == " + std::to_string(sizeof(type)) + ");\n"; \
    result += "static_assert(alignof(ascend_deepep::" #type ") == " + std::to_string(alignof(type)) + ");\n"
#define DEEPEP_DIRECT_ABI_FIELD(type, field)                                                 \
    result += "static_assert(__builtin_offsetof(ascend_deepep::" #type ", " #field ") == " + \
              std::to_string(offsetof(type, field)) + ");\n"
    DEEPEP_DIRECT_ABI_TYPE(DispatchTilingData);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, rank);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, world_size);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, num_tokens);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, hidden);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, num_topk);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, num_experts);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, num_max_tokens_per_rank);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, expert_alignment);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, block_dim);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, has_topk_weights);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, has_x_scales);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, cached_mode);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, num_channels);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, rank_stride);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, expert_stride);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, channel_stride);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, num_local_experts);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, x_element_bytes);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, scale_element_bytes);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, num_scales);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, num_output_tokens);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, needs_notify_entry_barrier);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, needs_x_scales_pack);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, reserved);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, scale_token_stride);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, scale_hidden_stride);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, notify_workspace_offset);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, recv_channel_prefix_offset);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, x_buffer_offset);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, src_idx_buffer_offset);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, topk_idx_buffer_offset);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, topk_weights_buffer_offset);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, x_scales_buffer_offset);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, x_scales_input_buffer_offset);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, udma_completion_signal_offset);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, dispatch_workspace_bytes);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, udma_peer_order);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, udma_ranks_per_group);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, udma_peers_per_worker);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, udma_wqes_per_visit);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, udma_token_schedule);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, udma_workers_per_peer);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, udma_qps_per_worker);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, v2_rank_prefix);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, v2_expert_prefix);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, v2_src_metadata);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, v2_dst_slots);
    DEEPEP_DIRECT_ABI_FIELD(DispatchTilingData, udma_output_control);
#undef DEEPEP_DIRECT_ABI_FIELD
#undef DEEPEP_DIRECT_ABI_TYPE
    result += "static_assert(sizeof(ascend_deepep::topk_idx_t) == " + std::to_string(sizeof(topk_idx_t)) + ");\n";
    return result;
}

}  // namespace ascend_deepep::direct_jit
