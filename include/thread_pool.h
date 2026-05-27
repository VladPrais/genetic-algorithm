#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class thread_pool
{
	typedef std::function<void(void)> task_type;

	std::vector<std::thread> threads;
	std::condition_variable cond_var;
	std::queue<task_type> task_queue;
	std::atomic<bool> stop_cond;
	std::mutex mutex;

	void work(void)
	{
		while(true)
		{
			std::unique_lock<std::mutex> locker(mutex);

			cond_var.wait(locker, [this](){ return !this->task_queue.empty() || this->stop_cond; });

			if (task_queue.empty() && stop_cond)
				return ;

			task_type task = std::move(task_queue.front());
			task_queue.pop();

			locker.unlock();

			task();
		}
	}

	public:

	thread_pool(int threads_count): stop_cond(false)
	{
		for(int i = 0; i < threads_count; i++)
		{
			threads.push_back(std::thread([this](){ this->work(); }));
		}
	}

	~thread_pool(void)
	{
		stop_cond = true;
		cond_var.notify_all();

		for(auto &t: threads)
			t.join();
	}

	void push(task_type task)
	{
		{
			std::unique_lock<std::mutex> locker(mutex);
			task_queue.push(task);
		}
		cond_var.notify_one();
	}
};

#endif
