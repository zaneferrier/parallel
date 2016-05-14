#pragma once

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
class worker;

//================================================================================

template <typename R>
class work_queue
{
public:

    explicit work_queue(unsigned num_workers)
        : continue_(true)
    { 
        workers_.reserve(num_workers); 
        for(unsigned i = 0; i < num_workers; ++i) {
            workers_.emplace_back(*this);
        }
    }

    ~work_queue()
    {
        wait();
        stop();
    }

    void push(std::function<R()>&& func)
    {
        std::lock_guard<std::mutex> _(work_lock_);
        work_.push(std::move(func));
        work_available_.notify_one();
    }

    void stop()
    {
        continue_.store(false);
        work_available_.notify_all();
    }

    void wait()
    {
        for(auto&& w : workers_) { w.wait(); } 
    }

private:

    void push_result(R&& result)
    {
        std::lock_guard<std::mutex> _(result_lock_);
        results_.emplace_back(std::move_if_noexcept(result));
        std::cout << "Result : " << results_.back() << '\n';
    }

    std::queue<std::function<R()>> work_;
    std::condition_variable        work_available_; 
    std::mutex                     work_lock_;
    std::mutex                     result_lock_;
    std::vector<worker<R>>         workers_;
    std::vector<R>                 results_;
    std::atomic<bool>              continue_;

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
        while(parent_queue_.continue_.load()) {
            std::unique_lock<std::mutex> lock(parent_queue_.work_lock_);
            parent_queue_.work_available_.wait(
                lock, [this]() { return check_condition(); }
            );
            if(!parent_queue_.continue_) { return; }
            if(parent_queue_.work_.empty()) { continue; }
            std::function<R()> f = parent_queue_.work_.front();
            parent_queue_.work_.pop();
            lock.unlock();
            auto result = f();
            parent_queue_.push_result(std::move(result));
        }
    }


private:

    bool check_condition()
    {
        return parent_queue_.continue_ || !parent_queue_.work_.empty();
    }

    work_queue<R>&                 parent_queue_;
    std::thread                    thread_;
};

} // end namespace internal
} // end namespace parallel
} // end namespace experimental
