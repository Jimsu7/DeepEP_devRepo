#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ascend_deepep::direct_jit {

// These stable IDs are also used by the device source guards. Only payload
// Dispatch belongs to this registry; Notify, layout, Combine, Expand and
// Aggregate keep their original AOT launchers.
enum class Entry : uint32_t {
    DispatchMte = 8,
    DispatchUdma = 12,
};

enum class Transport { Mte, Udma };
enum class Category { Mte, Udma };

struct Descriptor {
    Entry entry;
    const char* source_path;  // Relative to the separately installed jit/direct include root.
    const char* symbol;
    uint32_t ub_bytes;
    Category category;
};

inline constexpr std::array<Descriptor, 2> descriptors{{
    {Entry::DispatchMte, "kernels/dispatch/dispatch.cpp", "v1_dispatch_kernel", 216 * 1024, Category::Mte},
    {Entry::DispatchUdma, "kernels/dispatch/dispatch_udma.cpp", "v1_dispatch_udma_kernel", 216 * 1024, Category::Udma},
}};

inline const Descriptor& descriptor(Entry entry)
{
    for (const auto& item : descriptors)
        if (item.entry == entry) return item;
    throw std::invalid_argument("Invalid Direct Dispatch JIT entry");
}

inline Transport parse_transport(std::string_view value)
{
    if (value == "mte") return Transport::Mte;
    if (value == "udma") return Transport::Udma;
    throw std::invalid_argument("Direct JIT transport must be mte or udma");
}

inline std::vector<Entry> entries_for_transport(Transport transport)
{
    if (transport != Transport::Mte && transport != Transport::Udma)
        throw std::invalid_argument("Invalid Direct JIT transport");
    const auto category = transport == Transport::Mte ? Category::Mte : Category::Udma;
    std::vector<Entry> entries;
    for (const auto& item : descriptors)
        if (item.category == category) entries.push_back(item.entry);
    return entries;
}

inline std::string source(Entry entry)
{
    const auto& item = descriptor(entry);
    // Each payload Dispatch source supplies its own SDK includes and exposes
    // exactly one device entry while excluding its legacy AOT Host launcher.
    return "#define DEEPEP_DIRECT_JIT_ENTRY " + std::to_string(static_cast<uint32_t>(entry)) + "\n#include <" +
           item.source_path + ">\n";
}

}  // namespace ascend_deepep::direct_jit
