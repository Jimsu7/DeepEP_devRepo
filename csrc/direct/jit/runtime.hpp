#pragma once

#include <string>
#include <vector>

namespace ascend_deepep::direct_jit {

// Configuration is immutable for the lifetime of the process. Loaded kernels
// belong to their ACL context, which must outlive every Buffer using them.
void initialize(const std::string& library_root, const std::vector<std::string>& include_dirs,
                const std::string& signature);

// Compile/load only the selected transport's payload Dispatch before any
// SHMEM session is created. Other operators retain their AOT entry points.
// Python coordinates preparation failures across ranks.
void prepare(const std::string& transport);
void ensure_prepared(const std::string& transport);
std::vector<std::string> prepared_entries();

}  // namespace ascend_deepep::direct_jit
