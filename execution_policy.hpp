#pragma once

#include <cstdint>
#include <memory>
#include <type_traits>
#include <typeinfo>

namespace experimental
{
namespace parallel 
{

//================================================================================

class sequential_execution_policy;
class parallel_execution_policy;
class parallel_vector_execution_policy;
class execution_policy;

void swap(sequential_execution_policy& p1, sequential_execution_policy& p2);
void swap(parallel_execution_policy& p1, parallel_execution_policy& p2);
void swap(parallel_vector_execution_policy& p1, parallel_vector_execution_policy& p2);
void swap(execution_policy& p1, execution_policy& p2);

//================================================================================

template <typename Policy>
struct is_execution_policy
    : std::integral_constant<bool, false>
{ };

template <>
struct is_execution_policy<sequential_execution_policy>
    : std::integral_constant<bool, true>
{ };

template <>
struct is_execution_policy<parallel_execution_policy>
    : std::integral_constant<bool, true>
{ };

template <>
struct is_execution_policy<parallel_vector_execution_policy>
    : std::integral_constant<bool, true>
{ };

template <>
struct is_execution_policy<execution_policy>
    : std::integral_constant<bool, true>
{ };

template <typename Policy>
constexpr bool is_execution_policy_v = is_execution_policy<Policy>::value;

//================================================================================

class sequential_execution_policy 
{ 
public:

    sequential_execution_policy() = default;

    void swap(sequential_execution_policy&) {  }
};

//================================================================================

class parallel_execution_policy 
{ 
public:

    parallel_execution_policy() = default;
    void swap(parallel_execution_policy&) { }
};

//================================================================================

class parallel_vector_execution_policy 
{ 
public:

    parallel_vector_execution_policy() = default;
    void swap(parallel_vector_execution_policy&) { }
};

//================================================================================

void swap(sequential_execution_policy& p1, sequential_execution_policy& p2)
{
    p1.swap(p2);
}

void swap(parallel_execution_policy& p1, parallel_execution_policy& p2)
{
    p1.swap(p2);
}

void swap(parallel_vector_execution_policy& p1, parallel_vector_execution_policy& p2)
{
    p1.swap(p2);
}

//================================================================================

constexpr sequential_execution_policy seq{};
constexpr parallel_execution_policy par{};
constexpr parallel_vector_execution_policy par_vec{};

//================================================================================

template <typename T, typename U>
constexpr bool is_same_v = std::is_same<T, U>::value;

//================================================================================

class execution_policy
{
public:

    template <typename T>
    execution_policy(
        const T& exec,
        typename std::enable_if<is_execution_policy_v<T>>::type* = 0
    )
        : which(to_policy_type<T>())
    { 
        (void)exec;
        construct();
    }

    ~execution_policy()
    {
        destroy();
    }

    execution_policy& operator=(const execution_policy& other) noexcept
    {
        if(&other != this) {
            destroy();
            which = other.which;
            construct();
            return *this;
        }
    }

    template <typename T>
    typename std::enable_if<is_execution_policy_v<T>, execution_policy&>::type
    operator=(const T& exec) noexcept
    {
        (void)exec;
        policy_type other_which = to_policy_type<T>();
        destroy();
        which = other_which;
        construct();
        return *this;
    }

    void swap(execution_policy& other) noexcept
    { 
        using std::swap;
        std::swap(which, other.which);
        std::swap(policy, other.policy);
    }

    const std::type_info& target_type() const noexcept
    {
        switch(which) {
            case policy_type::sequential: return typeid(seq); 
            case policy_type::parallel: return typeid(par); 
            case policy_type::vector: return typeid(par_vec); 
        }
        std::terminate();
    }

    template <typename Policy>
    typename std::enable_if<is_execution_policy_v<Policy>, Policy*>::type
    get() noexcept
    {
        if(typeid(Policy) == target_type()) { 
            return reinterpret_cast<Policy*>(&policy);
        }
        return nullptr;
    }

    template <typename Policy>
    typename std::enable_if<is_execution_policy_v<Policy>, const Policy*>::type
    get() const noexcept
    {
        if(typeid(Policy) == target_type()) { 
            return reinterpret_cast<const Policy*>(&policy);
        }
        return nullptr;
    }

private:

    void construct()
    {
        switch(which) {
            case policy_type::sequential:
                new (static_cast<void*>(&policy)) sequential_execution_policy;
                break;
            case policy_type::parallel:
                new (static_cast<void*>(&policy)) parallel_execution_policy;
                break;
            case policy_type::vector:
                new (static_cast<void*>(&policy)) parallel_vector_execution_policy;
                break;
            default:
                std::terminate(); 
        }
    }

    void destroy()
    {
        switch(which) {
            case policy_type::sequential:
                reinterpret_cast<sequential_execution_policy*>(&policy)->
                    ~sequential_execution_policy();
                break;
            case policy_type::parallel:
                reinterpret_cast<parallel_execution_policy*>(&policy)->
                    ~parallel_execution_policy();
                break;
            case policy_type::vector:
                reinterpret_cast<parallel_vector_execution_policy*>(&policy)->
                    ~parallel_vector_execution_policy();
                break;
            default:
                std::terminate();
        }
    }

    enum class policy_type 
        : std::uint8_t
    { sequential, parallel, vector };
    
    policy_type which;

    std::aligned_union_t<
        1,
        sequential_execution_policy, 
        parallel_execution_policy,
        parallel_vector_execution_policy
    > policy;

    template <typename T>
    constexpr policy_type to_policy_type()
    {
         return
             is_same_v<T, sequential_execution_policy> 
                ? policy_type::sequential 
                : is_same_v<T, parallel_execution_policy> 
                    ? policy_type::parallel 
                    : policy_type::vector;
    }
};

void swap(execution_policy p1, execution_policy& p2) noexcept
{
    p1.swap(p2);    
}

} // end namespace parallel
} // end namespace experimental
