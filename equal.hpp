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

// Somewhat arbitrary size of a "small type". Anything that is this size
constexpr std::size_t small_type = 16;

template <typename T>
constexpr bool is_arithmetic_v = std::is_arithmetic<T>::value;

template <typename T>
constexpr bool is_small = is_arithmetic_v<T>|| sizeof(T) <= small_type; 

template <typename T>
constexpr bool is_simple = std::is_nothrow_copy_constructible<T>::value &&
                           !std::has_virtual_destructor<T>::value;

template <typename T>
constexpr bool by_value = is_small<T> && is_simple<T>;

template <typename T>
constexpr bool is_arithmetic_v = std::is_arithmetic<T>::value;

template <typename IterType>
using pass_type = 
    typename std::conditional<
        by_value<std::decay_t<typename std::iterator_traits<IterType>::value_type>>,
        typename std::iterator_traits<IterType>::value_type,
        const typename std::iterator_traits<IterType>::reference
    >::type;

// Find the lowest common denominator of the iterator types.
// Further, amke sure the most derived type is random_access_iterator_tag.
// This is because C++17 introduces Contiguous iterators, however,
// in the functions we want these to take the random_access_iterator_tag
// overloads.
// (Note: I should make something to detect contiguous iterators for
// vector overloads).
template <typename InputIt1, typename InputIt2>
using common_iter_type = 
    std::common_type_t<
        typename std::iterator_traits<InputIt1>::iterator_category,
        typename std::iterator_traits<InputIt2>::iterator_category,
        std::random_access_iterator_tag
    >;

//================================================================================
//=======================Sequential Execution Policy==============================
//================================================================================

// Sequential execution policy: simply forward to std::equal algorithm.
// This doesn't have any separate overloads for different iterator types.
template <typename InputIt1, typename InputIt2>
bool equal_impl(
    sequential_execution_policy, 
    InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt2 end2
)
{
    return std::equal(begin1, end1, begin2, end2);
}

template <typename InputIt1, typename InputIt2, typename BinaryPredicate>
bool equal_impl(
    sequential_execution_policy, 
    InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt2 end2,
    BinaryPredicate binary_pred
)
{
    return std::equal(begin1, end1, begin2, end2, binary_pred);
}

//================================================================================
//========================Parallel Execution Policy===============================
//================================================================================

// Parallel exceution policy for non-random access iterators doesn't really make
// any sense, since performing any splitting of the input will take O(n) time.
template <
    typename InputIt1, typename InputIt2, 
    typename IteratorTag, typename BinaryPredicate
>
bool equal_impl(
    parallel_execution_policy, 
    InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt2 end2,
    IteratorTag, BinaryPredicate binary_pred 
)
{
    return std::equal(begin1, end1, begin2, end2, binary_pred);
}

template <typename InputIt1, typename InputIt2, typename Predicate>
bool equal_impl(
    parallel_execution_policy, 
    InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt2 end2,
    std::random_access_iterator_tag, Predicate pred 
)
{
    const static unsigned hc = 2 * get_hardware_concurrency_or_default();
    const auto size = std::distance(begin1, end1);

    if(size != std::distance(begin2, end2)) { return false; }

    const auto chunk_size = static_cast<std::size_t>(size) / hc;
    std::vector<std::future<void>> tasks;
    tasks.reserve(hc);

    std::atomic<bool> are_same{true};
    
    for(auto i = 0U; i < hc; ++i) {
        const unsigned i_chunk = i * chunk_size;
        const unsigned next_i_chunk = i_chunk + chunk_size; 
        tasks.emplace_back(
            std::async(
                std::launch::async,
                [begin1, begin2, i_chunk, next_i_chunk, pred, &are_same] { 
                auto begin = begin1 + i_chunk;
                auto end = begin1 + next_i_chunk;
                auto begin2nd = begin2 + i_chunk;
                while(begin != end && are_same.load(std::memory_order_relaxed)) 
                {
                    if(!pred(*begin, *begin2nd)) { 
                        are_same.store(false, std::memory_order_relaxed); 
                        return; 
                    }
                    ++begin; ++begin2nd;
                }
            })
        );
    }

    for(auto&& task : tasks) { task.get(); }
    return are_same;
}

template <typename InputIt1, typename InputIt2, typename IteratorTag>
bool equal_impl(
    parallel_execution_policy, 
    InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt2 end2,
    IteratorTag
)
{
    using pass_type_first = pass_type<InputIt1>;
    using pass_type_second = pass_type<InputIt2>;

    return 
        std::equal(
           begin1, end1, begin2, end2, 
           [](pass_type_first v1, pass_type_second v2) { return v1 == v2; }
        );
}

// If a predicate isn't specified, generate one and use the above implementation.
template <typename InputIt1, typename InputIt2>
bool equal_impl(
    parallel_execution_policy pep, 
    InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt2 end2,
    std::random_access_iterator_tag tag
)
{
    using pass_type_first = pass_type<InputIt1>;
    using pass_type_second = pass_type<InputIt2>;

    return equal_impl(
        pep, begin1, end1, begin2, end2, tag, 
        [](pass_type_first v1, pass_type_second v2) { return v1 == v2; }
    );
}


//================================================================================
//=====================Parallel Vector Execution Policy===========================
//================================================================================

template <typename InputIt1, typename InputIt2, typename IteratorTag>
bool equal_impl(
    parallel_vector_execution_policy pvep, 
    InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt2 end2,
    IteratorTag tag
)
{ 
    using pass_type_first = pass_type<InputIt1>;
    using pass_type_second = pass_type<InputIt2>;

    return equal_impl(
        pvep, begin1, end1, begin2, end2, tag, 
        [](pass_type_first v1, pass_type_second v2) { return v1 == v2; }
    );
} 

template <
    typename InputIt1, typename InputIt2, 
    typename IteratorTag, typename BinaryPredicate
>
bool equal_impl(
    parallel_vector_execution_policy, 
    InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt2 end2,
    IteratorTag, BinaryPredicate binary_pred 
)
{ 
    // Not yet implemented
    std::terminate();
}

// Vectorized operations really only make sense on basic types using
// a simple equals. This should probably be in the overload where
// no predicate is specified.
template <typename InputIt1, typename InputIt2, typename BinaryPredicate>
bool equal_impl(
    parallel_vector_execution_policy, 
    InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt2 end2,
    std::random_access_iterator_tag, BinaryPredicate binary_pred 
)
{ 
    if(is_arithmetic_v<std::decay_t<decltype(*begin1)>> &&
       is_arithmetic_v<std::decay_t<decltype(*begin2)>>)
    {
        // Actually perform vectorized equal.
        // Not yet implemented
    }
    // More difficult with compound types, leave it sequential.
    return std::equal(begin1, end1, begin2, end2, binary_pred);
}

//================================================================================

template <typename InputIt1, typename InputIt2>
bool equal_impl(
    parallel_execution_policy pep, 
    InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt2 end2
)
{
    using common_type = common_iter_type<InputIt1, InputIt2>;
    return equal_impl(pep, begin1, end1, begin2, end2, common_type{});
}

template <typename InputIt1, typename InputIt2>
bool equal_impl(
    parallel_vector_execution_policy pvep, 
    InputIt1 begin1, InputIt1 end1, InputIt2 begin2, InputIt2 end2
)
{
    using common_type = common_iter_type<InputIt1, InputIt2>;
    return equal_impl(pvep, begin1, end1, begin2, end2, common_type{});
}

} // end namespace internal 

//================================================================================

template <typename T>
using basic_type = 
    typename std::remove_reference_t<
        typename std::remove_cv<T>
    >;

template <typename ExecutionPolicy, typename InputIt1, typename InputIt2>
bool equal(
    ExecutionPolicy&& policy, InputIt1 begin1, InputIt1 end1,
    InputIt2 begin2, InputIt2 end2,
    typename std::enable_if<is_execution_policy_v<std::decay_t<ExecutionPolicy>>>::type* = 0
)
{ 
    return internal::equal_impl(policy, begin1, end1, begin2, end2);
}

template <typename ExecutionPolicy, typename InputIt1, typename InputIt2, typename Predicate>
bool equal(
    ExecutionPolicy&& policy, InputIt1 begin1, InputIt1 end1,
    InputIt2 begin2, InputIt2 end2, Predicate pred,
    typename std::enable_if<is_execution_policy_v<std::decay_t<ExecutionPolicy>>>::type* = 0
)
{ 
    return internal::equal_impl(policy, begin1, end1, begin2, end2, pred);
}

template <typename InputIt1, typename InputIt2>
bool equal(
    execution_policy policy, InputIt1 begin1, InputIt1 end1,
    InputIt2 begin2, InputIt2 end2
)
{ 
    auto func = [begin1, end1, begin2, end2](auto policy)
                { return internal::equal_impl(policy, begin1, end1, begin2, end2); };
    return internal::dispatch(policy, func);
}

template <typename InputIt1, typename InputIt2, typename Predicate>
bool equal(
    execution_policy policy, InputIt1 begin1, InputIt1 end1,
    InputIt2 begin2, InputIt2 end2, Predicate pred
)
{ 
    auto func = [begin1, end1, begin2, end2, pred](auto policy)
                { return internal::equal_impl(policy, begin1, end1, begin2, end2, pred); };
    return internal::dispatch(policy, func);
}

} // end namespace parallel
} // end namespace experimental

