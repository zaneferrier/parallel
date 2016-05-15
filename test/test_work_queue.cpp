#include "work_queue.hpp"

#include <cassert>
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
        []() { return fib(41); },
        []() { return fib(30); },
        []() { return fib(41); },
        []() { return fib(38); },
        []() { return fib(32); },
        []() { return fib(39); },
        []() { return fib(40); }
    };

    for(auto&& f : func_test) {
        wq.push(std::move(f));
    }

    wq.shutdown();

    work_queue<void> void_queue(4);
    std::atomic<int> counter{0};
    for(auto i = 0; i < 10; ++i) {
        void_queue.push([&counter]() { ++counter; });
    }

    void_queue.shutdown();

    std::cout << "Counter: " << counter << '\n';
    assert(counter == 10);
}

