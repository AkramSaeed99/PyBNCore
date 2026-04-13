#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace bncore {

// Lightweight thread pool for batch inference.
// Threads are created once and reused across evaluate() calls,
// eliminating the ~50-200us per-call overhead of std::async.
class ThreadPool {
public:
  explicit ThreadPool(std::size_t num_threads)
      : stop_(false) {
    workers_.reserve(num_threads);
    for (std::size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this] { worker_loop(); });
    }
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto &w : workers_)
      w.join();
  }

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;

  // Submit a batch of tasks and wait for all to complete.
  // More efficient than individual submits for fork-join patterns.
  void run_batch(std::function<void(std::size_t)> task,
                 std::size_t num_tasks) {
    if (num_tasks == 0) return;
    if (num_tasks == 1) {
      task(0);
      return;
    }

    std::atomic<std::size_t> remaining{num_tasks};

    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (std::size_t i = 0; i < num_tasks; ++i) {
        tasks_.push([&task, &remaining, i] {
          task(i);
          remaining.fetch_sub(1, std::memory_order_release);
        });
      }
    }
    cv_.notify_all();

    // Spin-wait briefly then yield — tasks are typically short
    while (remaining.load(std::memory_order_acquire) > 0) {
      std::this_thread::yield();
    }
  }

  std::size_t num_threads() const { return workers_.size(); }

private:
  void worker_loop() {
    for (;;) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
        if (stop_ && tasks_.empty()) return;
        task = std::move(tasks_.front());
        tasks_.pop();
      }
      task();
    }
  }

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_;
};

} // namespace bncore
