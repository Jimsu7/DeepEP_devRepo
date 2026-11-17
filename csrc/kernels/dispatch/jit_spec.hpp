// Copyright (c) 2026, Lu Lu
#pragma once

#include <array>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>

namespace deepep::dispatch_jit {
// SDK-free specialization identity/source generation can be checked on a host
// without CANN; compilation/loading is implemented in jit_launcher.hpp.
struct DispatchSpec {
    uint32_t world, hidden, topk;
    bool fp8, with_weights, profile;
    bool operator==(const DispatchSpec&) const = default;

    std::array<uint32_t, 6> identity() const
    {
        return {world, hidden, topk, uint32_t(fp8), uint32_t(with_weights), uint32_t(profile)};
    }

    void validate() const
    {
        if (world < 2 || world > 512 || (world & (world - 1)) || hidden == 0 || hidden > 7168 ||
            hidden % (fp8 ? 128U : 16U) || topk == 0 || topk > 16)
            throw std::invalid_argument("Invalid dispatch specialization (EP2..512, H<=7168 aligned, K1..16)");
    }
};
inline const char* tag(const DispatchSpec& spec)
{
    return spec.profile ? "dispatch_profile" : "dispatch";
}
inline std::string source(const DispatchSpec& spec)
{
    spec.validate();
    // An explicit Ascend global wrapper exposes one logical ELF kernel entry.
    return std::format(R"(#include <kernels/dispatch/dispatch_impl.hpp>
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void dispatch_jit_kernel(
    GM_ADDR workspace, GM_ADDR destinations, GM_ADDR forward, GM_ADDR counts,
    GM_ADDR weights, GM_ADDR scales, ascend_deepep::DispatchTiling runtime)
{{
    util_set_ffts_config(0U);
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
    ascend_deepep::Dispatch<{}, {}, {}, {}, {}, {}>(
        workspace, destinations, forward, counts, weights, scales, runtime);
#endif
}}
)",
                       spec.world, spec.hidden, spec.topk, spec.fp8 ? "true" : "false",
                       spec.with_weights ? "true" : "false", spec.profile ? "true" : "false");
}
}  // namespace deepep::dispatch_jit
