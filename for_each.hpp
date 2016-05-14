#pragma once

#include <algorithm>
#include <future>
#include <iterator>
#include <thread>
#include <type_traits>
#include <vector>

#include "execution_policy.hpp"
#include "dispatch.hpp"

namespace experimental
{
namespace parallel
{
namespace internal
{

//================================================================================

template <typename InputIt, typename Func>
void for_each_impl(
    sequential_execution_policy, InputIt begin, InputIt end, Func f
)
{
    std::for_each(begin, end, f);
}

//================================================================================

template <typename InputIt, typename Func>
void for_each_impl(
    parallel_execution_policy, InputIt begin, InputIt end, Func f,
    typename std::enable_if<
        std::is_same<
            typename std::iterator_traits<InputIt>::iterator_category,
            std::random_access_iterator_tag
        >::value 
    >::type* = 0
)
{
    const static unsigned hc = 2 * get_hardware_concurrency_or_default();
    const auto size = std::distance(begin, end);
    const auto chunk_size = static_cast<std::size_t>(size) / hc;
    std::vector<std::future<void>> tasks;

    tasks.reserve(hc);
    
    for(auto i = 0U; i < hc; ++i) {
        const unsigned i_chunk = i * chunk_size;
        const unsigned next_i_chunk = i_chunk + chunk_size; 
        tasks.emplace_back(
            std::async(
                std::launch::async,
                [begin, i_chunk, next_i_chunk, f] { 
                auto begin_chunk = begin + i_chunk;
                auto end_chunk = begin + next_i_chunk;
                while(begin_chunk != end_chunk) {
                    f(*begin_chunk);
                    ++begin_chunk; 
                }
            })
        );
    }

    for(auto&& task : tasks) { task.get(); }
}

template <typename InputIt, typename Func>
void for_each_impl(
    parallel_execution_policy, InputIt begin, InputIt end, Func f,
    typename std::enable_if<
        !std::is_same<
            typename std::iterator_traits<InputIt>::iterator_category,
            std::random_access_iterator_tag
        >::value 
    >::type* = 0
)
{
    std::for_each(begin, end, f);
}

//================================================================================

template <typename InputIt, typename Func>
void for_each_impl(
    parallel_vector_execution_policy, InputIt begin, InputIt end, Func f
)
{
    for_each_impl(par, begin, end, f);
}

//================================================================================

} // end namespace internal

//================================================================================

template <typename ExecutionPolicy, typename InputIt, typename Func>
void for_each(
    ExecutionPolicy&& policy, InputIt begin, InputIt end, Func func,
    typename std::enable_if<is_execution_policy_v<std::decay_t<ExecutionPolicy>>>::type* = 0
)
{ 
    internal::for_each_impl(policy, begin, end, func);
}

template <typename InputIt, typename Func>
void for_each(
    execution_policy policy, InputIt begin, InputIt end, Func func 
)
{ 
    auto f = [begin, end, func](auto policy)
             { return internal::for_each_impl(policy, begin, end, func); };
    return internal::dispatch(policy, f);
}

} // end namespace parallel
} // end namespace experimental

