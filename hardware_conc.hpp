#pragma once

#include <thread>

namespace experimental
{
namespace parallel
{

inline unsigned get_hardware_concurrency_or_default()
{
    const static unsigned default_hc = 1;
    const unsigned hc = std::thread::hardware_concurrency();
    return (hc > 0) ? hc : default_hc;
}

} // end namespace parallel
} // end namespace experimental
