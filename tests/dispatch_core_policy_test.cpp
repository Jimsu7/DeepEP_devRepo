#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include "kernels/dispatch/core_policy.hpp"
using deepep::dispatch_jit::select_block_count;
int main()
{
    static_assert(select_block_count(64U) == 64U);
    static_assert(select_block_count(128U) == 64U);
    // Exhaust the unsupported interval, including 32, 56 and the 63/64 edge.
    for (uint32_t available = 0U; available < 64U; ++available) {
        bool rejected = false;
        try {
            (void)select_block_count(available);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        assert(rejected);
    }
    for (uint32_t available : {64U, 65U, 72U, 128U, UINT32_MAX}) assert(select_block_count(available) == 64U);
    std::cout << "PASS original fixed-64 Dispatch capacity boundary\n";
}
