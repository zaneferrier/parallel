#pragma once

#include "execution_policy.hpp"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>

#include <iostream>

namespace experimental
{
namespace parallel
{
namespace internal
{

//================================================================================

template <typename R>
class work_queue;

template <typename R>
class worker;

template <>
class work_queue<void>;

template <>
class worker<void>;

//================================================================================

template <typename R>
class work_queue
{
public:

    explicit work_queue(unsigned num_workers)
        : shutdown_(false)
    { 
        workers_.reserve(num_workers); 
        for(unsigned i = 0; i < num_workers; ++i) {
            workers_.emplace_back(*this);
        }
    }

    ~work_queue()
    {
        wait();
        shutdown();
    }

    void push(std::function<R()>&& func)
    {
        std::lock_guard<std::mutex> _(work_lock_);
        work_.push(std::move(func));
        work_available_.notify_one();
    }

    void shutdown()
    {
        shutdown_.store(true);
        work_available_.notify_all();
        wait();
    }

    void wait()
    {
        for(auto&& w : workers_) { w.wait(); } 
    }

    bool has_exceptions() const
    {
        std::lock_guard<std::mutex> _(result_lock_);
        return exceptions_.size() > 0;
    }

    exception_list exceptions() const
    {
        std::lock_guard<std::mutex> _(result_lock_);
        return exceptions_;
    }

private:

    void push_result(R&& result)
    {
        std::lock_guard<std::mutex> _(result_lock_);
        results_.emplace_back(std::move_if_noexcept(result));
        std::cout << "Result : " << results_.back() << '\n';
    }

    void add_exception(std::exception_ptr eptr)
    {
        std::lock_guard<std::mutex> _(result_lock_);
        exceptions_.push_back(eptr);
    }

    std::queue<std::function<R()>> work_;
    std::condition_variable        work_available_; 
    std::mutex                     work_lock_;
    mutable std::mutex             result_lock_;
    std::vector<worker<R>>         workers_;
    std::vector<R>                 results_;
    std::atomic<bool>              shutdown_;
    exception_list                 exceptions_;

    friend class worker<R>;
};

//================================================================================

template <typename R>
class worker
{
public:

    worker(work_queue<R>& wq)
        : parent_queue_(wq)
    { 
        thread_ = std::thread([this]() { start(); }); 
    }

    ~worker()
    {
        wait();
    }

    worker& operator=(worker&&) = default;
    worker(worker&&) = default;

    void wait()
    {
        if(thread_.joinable()) {
           thread_.join();
        }
    }

    void start()
    {
        while(1) {
            std::unique_lock<std::mutex> lock(parent_queue_.work_lock_);
            parent_queue_.work_available_.wait(
                lock, [this]() { return check_condition(); }
            );
            if(parent_queue_.shutdown_ && parent_queue_.work_.empty()) { return; }
            if(parent_queue_.shutdown_) { lock.unlock(); shutdown(); return; }
            if(parent_queue_.work_.empty()) { continue; }
            std::function<R()> f = parent_queue_.work_.front();
            parent_queue_.work_.pop();
            lock.unlock();
            try {
                auto result = f();
                parent_queue_.push_result(std::move(result));
            } catch(...) {
                auto except = std::current_exception();
                parent_queue_.add_exception(except);
            }
        }
    }


private:

    void shutdown()
    {
        while(1) {
            std::unique_lock<std::mutex> lock(parent_queue_.work_lock_);
            if(parent_queue_.work_.empty()) { return; }
            std::function<R()> f = parent_queue_.work_.front();
            parent_queue_.work_.pop();
            lock.unlock();
            try {
                auto result = f();
                parent_queue_.push_result(std::move(result));
            } catch(...) {
                auto except = std::current_exception();
                parent_queue_.add_exception(except);
            }
        }
    }

    bool check_condition()
    {
        return parent_queue_.shutdown_ || !parent_queue_.work_.empty();
    }

    work_queue<R>& parent_queue_;
    std::thread    thread_;
};

//================================================================================

template <>
class work_queue<void>
{
public:

    explicit work_queue(unsigned num_workers)
        : shutdown_(false)
    {
        workers_.reserve(num_workers);
        for(unsigned i = 0; i < num_workers; ++i) {
            workers_.emplace_back(*this);
        }
    }

    ~work_queue()
    {
        wait();
        shutdown();
    }

    void push(std::function<void()>&& func)
    {
        std::lock_guard<std::mutex> _(work_lock_);
        work_.push(std::move(func));
        work_available_.notify_one();
    }

    void shutdown()
    {
        shutdown_.store(true);
        work_available_.notify_all();
        wait();
    }

    // This must be defined lower, as it depends on
    // worker<void>, which has only been forward declared
    // at this point.
    void wait();

    bool has_exceptions() const
    {
        std::lock_guard<std::mutex> _(exception_lock_);
        return exceptions_.size() > 0;
    }

    exception_list exceptions() const
    {
        std::lock_guard<std::mutex> _(exception_lock_);
        return exceptions_;
    }

private:

    void add_exception(std::exception_ptr eptr)
    {
        std::lock_guard<std::mutex> _(exception_lock_);
        exceptions_.push_back(eptr);
    }

    std::queue<std::function<void()>> work_;
    std::condition_variable           work_available_;
    std::mutex                        work_lock_;
    mutable std::mutex                exception_lock_;;
    std::vector<worker<void>>         workers_;
    std::atomic<bool>                 shutdown_;
    exception_list                    exceptions_;

    friend class worker<void>;
};

//================================================================================

template <>
class worker<void>
{
public:

    worker(work_queue<void>& wq)
        : parent_queue_(wq)
    {
        thread_ = std::thread([this]() { start(); });
    }

    ~worker()
    {
        wait();
    }

    worker& operator=(worker&&) = default;
    worker(worker&&) = default;

    void wait()
    {
        if(thread_.joinable()) {
           thread_.join();
        }
    }

    void start()
    {
        while(1) {
            std::unique_lock<std::mutex> lock(parent_queue_.work_lock_);
            parent_queue_.work_available_.wait(
                lock, [this]() { return check_condition(); }
            );
            if(parent_queue_.shutdown_ && parent_queue_.work_.empty()) { return; }
            if(parent_queue_.shutdown_) { lock.unlock(); shutdown(); return; }
            if(parent_queue_.work_.empty()) { continue; }
            std::function<void()> f = parent_queue_.work_.front();
            parent_queue_.work_.pop();
            lock.unlock();
            try {
                f();
            } catch(...) {
                auto except = std::current_exception();
                parent_queue_.add_exception(except);
            }
        }
    }


private:

    void shutdown()
    {
        while(1) {
            std::unique_lock<std::mutex> lock(parent_queue_.work_lock_);
            if(parent_queue_.work_.empty()) { return; }
            std::function<void()> f = parent_queue_.work_.front();
            parent_queue_.work_.pop();
            lock.unlock();
            try {
                f();
            } catch(...) {
                auto except = std::current_exception();
                parent_queue_.add_exception(except);
            }
        }
    }

    bool check_condition()
    {
        return parent_queue_.shutdown_ || !parent_queue_.work_.empty();
    }

    work_queue<void>& parent_queue_;
    std::thread       thread_;
};

//================================================================================

// This needs to be defined -after- worker<void> has been declared.
// might be a good idea to break this up into a header and .inc file
// to get around this issue.
void work_queue<void>::wait()
{
    for(auto&& w : workers_) { w.wait(); }
}

//================================================================================

} // end namespace internal
} // end namespace parallel
} // end namespace experimental
