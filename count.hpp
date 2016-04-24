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

template <typename InputIt, typename T>
typename std::iterator_traits<InputIt>::difference_type
count_impl(
    sequential_execution_policy, InputIt begin, InputIt end, const T& value
)
{
    std::count(begin, end, value);
}

template <typename InputIt, typename UnaryPredicate>
typename std::iterator_traits<InputIt>::difference_type
count_if_impl(
    sequential_execution_policy, InputIt begin, InputIt end, UnaryPredicate p
)
{
    std::count_if(begin, end, p);
}

//================================================================================

template <typename Iterator>
using enable_if_random = 
    typename std::enable_if<
        std::is_same<
            typename std::iterator_traits<Iterator>::iterator_category,
            std::random_access_iterator_tag
        >::value
    >::type;

template <typename Iterator>
using enable_if_not_random = 
    typename std::enable_if<
        !std::is_same<
            typename std::iterator_traits<Iterator>::iterator_category,
            std::random_access_iterator_tag
        >::value
    >::type;

//================================================================================

template <typename InputIt, typename Predicate>
typename std::iterator_traits<InputIt>::difference_type
count_impl_base(
    parallel_execution_policy, InputIt begin, InputIt end, Predicate p,
    enable_if_random<InputIt>* = 0 
)
{
    using return_type = typename std::iterator_traits<InputIt>::difference_type;
    using future_type = std::future<return_type>;

    const static unsigned hc = 2 * get_hardware_concurrency_or_default();
    const auto size = std::distance(begin, end);
    const auto chunk_size = static_cast<std::size_t>(size) / hc;
    std::vector<future_type> tasks;

    tasks.reserve(hc);
    
    for(auto i = 0U; i < hc; ++i) {
        const unsigned i_chunk = i * chunk_size;
        const unsigned next_i_chunk = i_chunk + chunk_size; 
        tasks.emplace_back(
            std::async(
                std::launch::async,
                [begin, i_chunk, next_i_chunk, p] { 
                return_type seen{0};
                auto begin_chunk = begin + i_chunk;
                auto end_chunk = begin + next_i_chunk;
                for(begin_chunk; begin_chunk != end_chunk; ++begin_chunk) {
                    if(p(*begin_chunk)) ++seen;
                }
                return seen;
            })
        );
    }

    return_type seen{0};
    for(auto&& task : tasks) { seen += task.get(); }
    return seen;
}

//================================================================================

template <typename InputIt, typename T>
typename std::iterator_traits<InputIt>::difference_type
count_impl(
    parallel_execution_policy pep, InputIt begin, InputIt end, const T& value, 
    enable_if_random<InputIt>* = 0
)
{
    return count_impl_base(pep, begin, end, 
        [&value](const T& input) { return input == value; });
}

template <typename InputIt, typename Predicate>
typename std::iterator_traits<InputIt>::difference_type
count_if_impl(
    parallel_execution_policy pep, InputIt begin, InputIt end, Predicate p,
    enable_if_random<InputIt>* = 0
)
{
    return count_impl_base(pep, begin, end, p);
}

template <typename InputIt, typename T> 
typename std::iterator_traits<InputIt>::difference_type
count_impl(
    parallel_execution_policy, InputIt begin, InputIt end, const T& value,
    enable_if_not_random<InputIt>* = 0    
)
{
    return std::count(begin, end, value);
}

template <typename InputIt, typename UnaryPredicate> 
typename std::iterator_traits<InputIt>::difference_type
count_if_impl(
    parallel_execution_policy, InputIt begin, InputIt end, UnaryPredicate p,
    enable_if_not_random<InputIt>* = 0
)
{
    return std::count_if(begin, end, p);
}

//================================================================================

template <typename InputIt, typename T>
typename std::iterator_traits<InputIt>::difference_type
count_impl(
    parallel_vector_execution_policy, InputIt begin, InputIt end, const T& value 
)
{
    return count_impl(par, begin, end, value);
}

template <typename InputIt, typename UnaryPredicate>
typename std::iterator_traits<InputIt>::difference_type
count_if_impl(
    parallel_vector_execution_policy, InputIt begin, InputIt end, UnaryPredicate p
)
{
    return count_if_impl(par, begin, end, p);
}

//================================================================================

} // end namespace internal

//================================================================================

template <typename ExecutionPolicy, typename InputIt, typename T>
typename std::iterator_traits<InputIt>::difference_type
count(
    ExecutionPolicy&& policy, InputIt begin, InputIt end, const T& value,
    typename std::enable_if<is_execution_policy_v<std::decay_t<ExecutionPolicy>>>::type* = 0
)
{ 
    internal::count_impl(policy, begin, end, value);
}

template <typename InputIt, typename T>
typename std::iterator_traits<InputIt>::difference_type
count(
    execution_policy policy, InputIt begin, InputIt end, const T& value 
)
{ 
    auto func = [begin, end, &value](auto policy)
             { return internal::count_impl(policy, begin, end, value); };
    return internal::dispatch(policy, func);
}

template <typename ExecutionPolicy, typename InputIt, typename UnaryPredicate>
typename std::iterator_traits<InputIt>::difference_type
count_if(
    ExecutionPolicy&& policy, InputIt begin, InputIt end, UnaryPredicate p,
    typename std::enable_if<is_execution_policy_v<std::decay_t<ExecutionPolicy>>>::type* = 0
)
{ 
    internal::count_if_impl(policy, begin, end, p);
}

template <typename InputIt, typename UnaryPredicate>
typename std::iterator_traits<InputIt>::difference_type
count_if(
    execution_policy policy, InputIt begin, InputIt end, UnaryPredicate p
)
{ 
    auto func = [begin, end, p](auto policy)
             { return internal::count_if_impl(policy, begin, end, p); };
    return internal::dispatch(policy, func);
}

} // end namespace parallel
} // end namespace experimental

