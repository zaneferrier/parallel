#include "work_queue.hpp"

#include <cstdint>

std::uint64_t fib(unsigned n)
{
    if(n == 0) { return 0; }
    if(n == 1) { return 1; }
    return fib(n - 1) + fib(n - 2);
}

int main()
{
    using namespace experimental::parallel::internal;

    work_queue<std::uint64_t> wq(4);

    std::vector<std::function<std::uint64_t()>> func_test {
        []() { return fib(20); },
        []() { return fib(25); },
        []() { return fib(45); },
        []() { return fib(30); },
        []() { return fib(45); },
        []() { return fib(38); },
        []() { return fib(32); },
        []() { return fib(47); },
        []() { return fib(42); }
    };

    for(auto&& f : func_test) {
        wq.push(std::move(f));
    }

    wq.wait();
}

