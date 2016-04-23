#pragma once

#include <algorithm>
#include <atomic>
#include <future>
#include <iterator>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "dispatch.hpp"
#include "execution_policy.hpp"
#include "hardware_conc.hpp"

namespace experimental
{
namespace parallel 
{
namespace internal 
{

//================================================================================
//=======================Sequential Execution Policy==============================
//================================================================================

// Sequential execution policy: simply forward to std::algorithm variants.
// This doesn't have any separate overloads for different iterator types.

template <typename InputIt, typename IterTag, typename Predicate> 
bool any_of_impl(
    sequential_execution_policy, InputIt begin, InputIt end, IterTag, Predicate pred
) 
{ return std::any_of(begin, end, pred); }

template <typename InputIt, typename IterTag, typename Predicate> 
bool all_of_impl(
    sequential_execution_policy, InputIt begin, InputIt end, IterTag, Predicate pred
) 
{ return std::all_of(begin, end, pred); }

template <typename InputIt, typename IterTag, typename Predicate> 
bool none_of_impl(
    sequential_execution_policy, InputIt begin, InputIt end, IterTag, Predicate pred
) 
{ return std::none_of(begin, end, pred); }

//================================================================================
//========================Parallel Execution Policy===============================
//================================================================================

// Base template that will handle the implementation of all 3 algorithms.

template <typename InputIt, typename Predicate, bool InitialResult>
bool any_all_none_impl(
    parallel_execution_policy, InputIt begin, InputIt end,
    std::random_access_iterator_tag, Predicate pred 
)
{
    const static unsigned hc = 2 * get_hardware_concurrency_or_default();
    const auto size = std::distance(begin, end);
    const auto chunk_size = static_cast<std::size_t>(size) / hc;
    const bool initial = InitialResult;
    std::vector<std::future<void>> tasks;

    std::atomic<bool> result{InitialResult};
    std::atomic<bool> continue_search{true};
    tasks.reserve(hc);
    
    for(auto i = 0U; i < hc; ++i) {
        const unsigned i_chunk = i * chunk_size;
        const unsigned next_i_chunk = i_chunk + chunk_size; 
        tasks.emplace_back(
            std::async(
                std::launch::async,
                [begin, i_chunk, next_i_chunk, pred, &result, &continue_search] { 
                auto begin_chunk = begin + i_chunk;
                auto end_chunk = begin + next_i_chunk;
                while(begin_chunk != end_chunk && 
                      continue_search.load(std::memory_order_relaxed)
                ) 
                {
                    if(pred(*begin_chunk) != initial) { 
                        result.store(!InitialResult, std::memory_order_relaxed); 
                        continue_search.store(false, std::memory_order_relaxed);
                        return; 
                    }
                    ++begin_chunk; 
                }
            })
        );
    }

    for(auto&& task : tasks) { task.get(); }
    return result;
}

//--------------------------------------------------------------------------------

template <typename InputIt, typename Predicate>
bool any_of_impl(
    parallel_execution_policy pep, InputIt begin, InputIt end,
    std::random_access_iterator_tag tag, Predicate pred 
)
{
    return any_all_none_impl<InputIt, Predicate, false>(
               pep, begin, end, tag, pred
           );
}

template <typename InputIt, typename Predicate>
bool all_of_impl(
    parallel_execution_policy pep, InputIt begin, InputIt end,
    std::random_access_iterator_tag tag, Predicate pred 
)
{
    return any_all_none_impl<InputIt, Predicate, true>(
               pep, begin, end, tag, pred
           );
}

template <typename InputIt, typename Predicate>
bool none_of_impl(
    parallel_execution_policy pep, InputIt begin, InputIt end,
    std::random_access_iterator_tag tag, Predicate pred 
)
{
    return !any_of_impl(pep, begin, end, tag, pred);
}

//--------------------------------------------------------------------------------

// Parallel execution policy but non-random access iterators, just
// forward this to the normal (sequential) std::algorithm functions.

template <typename InputIt, typename IterTag, typename Predicate>
bool any_of_impl(
    parallel_execution_policy pep, InputIt begin, InputIt end,
    IterTag, Predicate pred 
)
{ return std::any_of(begin, end, pred); }

template <typename InputIt, typename IterTag, typename Predicate>
bool all_of_impl(
    parallel_execution_policy pep, InputIt begin, InputIt end,
    std::random_access_iterator_tag tag, Predicate pred 
)
{ return std::all_of(begin, end, pred); }

template <typename InputIt, typename IterTag, typename Predicate>
bool none_of_impl(
    parallel_execution_policy pep, InputIt begin, InputIt end,
    std::random_access_iterator_tag tag, Predicate pred 
)
{ return std::none_of(begin, end, pred); }

//================================================================================
//=====================Parallel Vector Execution Policy===========================
//================================================================================

// These simply forward to the exact same functions as the parallel implementation
// for now.
// For actual vectorization, how would this work with regards to the predicate?

template <typename InputIt, typename Predicate>
bool any_of_impl(
    parallel_vector_execution_policy pep, InputIt begin, InputIt end,
    std::random_access_iterator_tag tag, Predicate pred 
)
{
    // Careful, note the swap from pep here to par (which is the global
    // parallel_execution_policy object).
    return any_all_none_impl<InputIt, Predicate, false>(
               par, begin, end, tag, pred
           );
}

template <typename InputIt, typename Predicate>
bool all_of_impl(
    parallel_vector_execution_policy pep, InputIt begin, InputIt end,
    std::random_access_iterator_tag tag, Predicate pred 
)
{
    return any_all_none_impl<InputIt, Predicate, true>(
               par, begin, end, tag, pred
           );
}

template <typename InputIt, typename Predicate>
bool none_of_impl(
    parallel_vector_execution_policy pep, InputIt begin, InputIt end,
    std::random_access_iterator_tag tag, Predicate pred 
)
{
    return !any_of_impl(par, begin, end, tag, pred);
}

//--------------------------------------------------------------------------------

// Parallel Vector execution policy but non-random access iterators, just
// forward this to the normal (sequential) std::algorithm functions.

template <typename InputIt, typename IterTag, typename Predicate>
bool any_of_impl(
    parallel_vector_execution_policy pep, InputIt begin, InputIt end,
    IterTag, Predicate pred 
)
{ return std::any_of(begin, end, pred); }

template <typename InputIt, typename IterTag, typename Predicate>
bool all_of_impl(
    parallel_vector_execution_policy pep, InputIt begin, InputIt end,
    std::random_access_iterator_tag tag, Predicate pred 
)
{ return std::all_of(begin, end, pred); }

template <typename InputIt, typename IterTag, typename Predicate>
bool none_of_impl(
    parallel_vector_execution_policy pep, InputIt begin, InputIt end,
    std::random_access_iterator_tag tag, Predicate pred 
)
{ return std::none_of(begin, end, pred); }

//================================================================================
//=============================Dispatch Functions=================================
//================================================================================

// These pull out the tag information from each iterator, and forward this on. 

template <typename ExecutionPolicy, typename InputIt, typename Predicate>
bool any_of_impl(
    ExecutionPolicy&& policy, InputIt begin, InputIt end, Predicate pred
)
{
    using iter_type = typename std::iterator_traits<InputIt>::iterator_category;
    return any_of_impl(policy, begin, end, iter_type{}, pred);
}

template <typename ExecutionPolicy, typename InputIt, typename Predicate>
bool all_of_impl(
    ExecutionPolicy&& policy, InputIt begin, InputIt end, Predicate pred
)
{
    using iter_type = typename std::iterator_traits<InputIt>::iterator_category;
    return all_of_impl(policy, begin, end, iter_type{}, pred);
}

template <typename ExecutionPolicy, typename InputIt, typename Predicate>
bool none_of_impl(
    ExecutionPolicy&& policy, InputIt begin, InputIt end, Predicate pred
)
{
    using iter_type = typename std::iterator_traits<InputIt>::iterator_category;
    return none_of_impl(policy, begin, end, iter_type{}, pred);
}

} // end namespace internal

//================================================================================

template <typename ExecutionPolicy, typename InputIt, typename Predicate>
bool any_of(
    ExecutionPolicy&& policy, InputIt begin, InputIt end, Predicate pred,
    typename std::enable_if<is_execution_policy_v<std::decay_t<ExecutionPolicy>>>::type* = 0
)
{ 
    internal::any_of_impl(policy, begin, end, pred);
}

template <typename ExecutionPolicy, typename InputIt, typename Predicate>
bool all_of(
    ExecutionPolicy&& policy, InputIt begin, InputIt end, Predicate pred,
    typename std::enable_if<is_execution_policy_v<std::decay_t<ExecutionPolicy>>>::type* = 0
)
{ 
    internal::all_of_impl(policy, begin, end, pred);
}

template <typename ExecutionPolicy, typename InputIt, typename Predicate>
bool none_of(
    ExecutionPolicy&& policy, InputIt begin, InputIt end, Predicate pred,
    typename std::enable_if<is_execution_policy_v<std::decay_t<ExecutionPolicy>>>::type* = 0
)
{ 
    internal::none_of_impl(policy, begin, end, pred);
}

template <typename InputIt, typename Predicate>
bool any_of(
    execution_policy policy, InputIt begin, InputIt end, Predicate pred
)
{ 
    auto func = [begin, end, pred](auto policy)
                { return internal::any_of_impl(policy, begin, end, pred); };
    return internal::dispatch(policy, func);
}

template <typename InputIt, typename Predicate>
bool all_of(
    execution_policy policy, InputIt begin, InputIt end, Predicate pred
)
{ 
    auto func = [begin, end, pred](auto policy)
                { return internal::all_of_impl(policy, begin, end, pred); };
    return internal::dispatch(policy, func);
}

template <typename InputIt, typename Predicate>
bool none_of(
    execution_policy policy, InputIt begin, InputIt end, Predicate pred
)
{ 
    auto func = [begin, end, pred](auto policy)
                { return internal::none_of_impl(policy, begin, end, pred); };
    return internal::dispatch(policy, func);
}

} // end namespace parallel
} // end namespace experimental

