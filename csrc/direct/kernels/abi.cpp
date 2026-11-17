#include "common/launch.h"
#include "common/types.h"

// This Host-only marker belongs to the AOT support library, not a Dispatch
// device translation unit. The distinct name also rejects older libraries
// that still contain AOT payload Dispatch and would shadow the JIT launchers.
extern "C" ASCEND_DEEPEP_EXPORT uint64_t direct_aot_dispatch_split_abi_1()
{
    return (uint64_t(sizeof(ascend_deepep::DispatchTilingData)) << 32) |
           uint64_t(sizeof(ascend_deepep::CombineTilingData));
}
