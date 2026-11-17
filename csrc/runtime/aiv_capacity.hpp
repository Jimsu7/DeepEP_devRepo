// Copyright (c) 2026, Lu Lu

#pragma once

#include <acl/acl_rt.h>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace deepep {

// A SoC name does not establish resident AIV capacity: an Ascend950 device
// can expose 56 AIVs. Every kernel using an all-block barrier must respect it.
// Do not cache this query: the calling thread's current device can change.
// This guards hardware capacity; it does not infer optional stream/core limits.
inline uint32_t QueryAvailableVectorCores()
{
    int32_t device = -1;
    auto status = aclrtGetDevice(&device);
    if (status != 0 || device < 0) {
        throw std::runtime_error(
            "Cannot query current ACL device for Vector-core capacity (status=" + std::to_string(status) + ")");
    }
    int64_t available = 0;
    status = aclrtGetDeviceInfo(static_cast<uint32_t>(device), ACL_DEV_ATTR_VECTOR_CORE_NUM, &available);
    if (status != 0 || available <= 0 || available > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Cannot query available Vector cores on device " + std::to_string(device) +
                                 " (status=" + std::to_string(status) + ", value=" + std::to_string(available) + ")");
    }
    return static_cast<uint32_t>(available);
}

}  // namespace deepep
