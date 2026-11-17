// Copyright (c) 2026, Lu Lu
#pragma once

#include <cstdint>
#include <stdexcept>

namespace deepep::dispatch_jit {
// The original Mesh/Clos HT Dispatch uses physical blocks 0..63. Its rear
// 32 blocks own weights/scales, so a smaller launch cannot preserve the ABI.
// This hardware check is independent of Notify and Direct kernel policies.
inline constexpr uint32_t select_block_count(uint32_t available)
{
    if (available < 64U) throw std::invalid_argument("Mesh/Clos HT Dispatch requires at least 64 available AIV cores");
    return 64U;
}
}  // namespace deepep::dispatch_jit
