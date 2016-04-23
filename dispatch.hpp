#pragma once

#include "execution_policy.hpp"

#include <typeinfo>

namespace experimental
{
namespace parallel
{
namespace internal
{

template <typename Func, typename... Args>
auto dispatch(execution_policy p, Func f, Args&&... args)
{
    if(p.target_type() == typeid(seq)) {
        auto * pol = p.get<sequential_execution_policy>();
        return f(*pol, std::forward<Args>(args)...);
    }
    else if(p.target_type() == typeid(par)) {
        auto* pol = p.get<parallel_execution_policy>();
        return f(*pol, std::forward<Args>(args)...);
    }
    else if(p.target_type() == typeid(par_vec)) {
        auto* pol = p.get<parallel_vector_execution_policy>();
        return f(*pol, std::forward<Args>(args)...);
    }
    std::terminate();
}

} // end namespace internal
} // end namespace parallel
} // end namespace experimental
