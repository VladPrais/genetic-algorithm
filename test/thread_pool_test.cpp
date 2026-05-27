#include <chrono>
#include <iostream>
#include "thread_pool.h"

void pause(void)
{
	std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

int main(void)
{
	int thread_count = std::thread::hardware_concurrency();
	std::cout << "Threads count " << thread_count << std::endl;

	auto t1 = std::chrono::steady_clock::now();
	
	for(int i = 0; i < thread_count; i++)
	{
		pause();
	}

	auto t2 = std::chrono::steady_clock::now();
	std::chrono::duration<double> r = t2 - t1;
	std::cout << r.count() << std::endl;

	t1 = std::chrono::steady_clock::now();

	{
		thread_pool pool(thread_count);
		auto task = [](){ pause(); };
		for(int i = 0; i < thread_count; i++)
		{
			pool.push(task);
		}
	}

	t2 = std::chrono::steady_clock::now();
	r = t2 - t1;
	std::cout << r.count() << std::endl;

	return 0;
}
