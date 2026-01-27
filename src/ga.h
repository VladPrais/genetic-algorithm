#ifndef __GA_H__
#define __GA_H__

#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "operators_ga.h"
#include "../../thread-pool/src/thread_pool.h"

namespace ga {

template <typename GeneType, typename FitnessType>
struct BaseIndividual
{
	GeneType genes;
	FitnessType fitness;
	bool valid;

	BaseIndividual():
		valid(false)
	{	}

	BaseIndividual(GeneType &genes):
		valid(false),
		genes(genes)
	{	}

	void evaluate(std::function<FitnessType(BaseIndividual<GeneType, FitnessType> &i)> fit_func)
	{
		fitness = fit_func(*this);
		valid = true;
	}

	bool is_valid(void) const
	{
		return valid;
	}
};

template <typename GeneType, typename FitnessType>
struct BaseGenetic
{
	using Individual = BaseIndividual<GeneType, FitnessType>;
	using Population = std::vector<Individual>;

	using Generator = std::function<Individual(std::mt19937&)>;
	using Comparator = std::function<bool(const Individual&, const Individual&)>;
	using Evaluate = std::function<FitnessType(Individual&)>;
	using StopCond = std::function<bool(Individual&)>;

	using Selection = std::function<void(typename Population::iterator, typename Population::iterator, Comparator, std::mt19937&)>;
	using Crossover = std::function<void(typename Population::iterator, typename Population::iterator, std::mt19937&)>;
	using Mutation = std::function<void(typename Population::iterator, typename Population::iterator, std::mt19937&)>;

	std::random_device rd;
	std::vector<std::mt19937> mt;

	int max_gen;
	int pop_size;
	int elite;

	Generator generator;
	Comparator comparator;
	Evaluate evaluate;
	StopCond stop_cond;

	Selection select;
	Crossover mate;
	Mutation mutate;

	bool output;

	BaseGenetic(Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Crossover mate, Mutation mut):
		max_gen(50),
		pop_size(100),
		elite(10),
		generator(gen),
		comparator(comp),
		evaluate(eval),
		stop_cond(stop),
		select(sel),
		mate(mate),
		mutate(mut),
		output(true)
	{
		config();
	}

	BaseGenetic(int max_gen, int pop_size, int elite, Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Crossover mate, Mutation mut):
		max_gen(max_gen),
		pop_size(pop_size),
		elite(elite),
		generator(gen),
		comparator(comp),
		evaluate(eval),
		stop_cond(stop),
		select(sel),
		mate(mate),
		mutate(mut),
		output(true)
	{
		config();
	}

	Individual operator()(void)
	{
		init_mt();

		Population population(pop_size);

		std::generate(population.begin(), population.end(), [this](){ return generator(mt[0]); });
		int evaluated_count = evaluate_pop(population.begin(), population.end());

		Individual best = get_best(population.begin(), population.end(), comparator);
		Individual worst = get_worst(population.begin(), population.end(), comparator);

		if (output)
		{
			print_head();
			print_stat(0, evaluated_count, worst.fitness, best.fitness);
		}

		if (stop_cond(best))
		{
			return best;
		}

		for(int iter = 1; iter <= max_gen; iter++)
		{
			selection_operator(population.begin(), population.end());
			crossover_operator(population.begin(), population.end());
			mutation_operator(population.begin(), population.end());
			evaluated_count = evaluate_pop(population.begin(), population.end());

			best = get_best(population.begin(), population.end(), comparator);
			worst = get_worst(population.begin(), population.end(), comparator);

			if (output)
			{
				print_stat(iter, evaluated_count, worst.fitness, best.fitness);
			}

			if(stop_cond(best))
			{
				return best;
			}
		}

		return best;
	}

	virtual int evaluate_pop(typename Population::iterator begin, typename Population::iterator end)
	{
		int count = 0;

		for(typename Population::iterator it = begin; it != end; it++)
		{
			if(!it->is_valid())
			{
				++count;
				it->evaluate(evaluate);
			}
		}

		return count;
	}

	virtual void selection_operator(typename Population::iterator begin, typename Population::iterator end)
	{
		select(begin, end, comparator, mt[0]);
	}

	virtual void crossover_operator(typename Population::iterator begin, typename Population::iterator end)
	{
		mate(begin, end, mt[0]);
	}

	virtual void mutation_operator(typename Population::iterator begin, typename Population::iterator end)
	{
		mutate(begin, end, mt[0]);
	}

	static void print_head(void)
	{
		char ch = '\t';
		std::cout << "iter" << ch << "evals" << ch << "low-fit" << ch << "high-fit" << std::endl;
	}

	static void print_stat(int iteration, int evaluated_count, FitnessType worst_val, FitnessType best_val)
	{
		char ch = '\t';
		std::cout << iteration << ch << evaluated_count << ch << worst_val << ch << best_val << std::endl;
	}

	static Individual get_best(typename Population::iterator begin, typename Population::iterator end, Comparator comp)
	{
		auto best = std::min_element(begin, end, comp);
		return *best;
	}

	static Individual get_worst(typename Population::iterator begin, typename Population::iterator end, Comparator comp)
	{
		auto worst = std::max_element(begin, end, comp);
		return *worst;
	}

	virtual void init_mt(void)
	{
		std::random_device rd;
		mt.push_back(std::mt19937(rd()));
	}

	void config(void)
	{
		bool underflow_max_gen = max_gen <= 0;
		bool underflow_pop_size = pop_size <= 0;
		bool underflow_elite = elite < 0;
		bool overflow_elite = elite >= pop_size;

		if(underflow_max_gen)
			throw std::underflow_error("Underflow max generations value " + std::to_string(max_gen));
		if(underflow_pop_size)
			throw std::underflow_error("Underflow population size value " + std::to_string(pop_size));
		if(underflow_elite)
			throw std::underflow_error("Underflow elite value " + std::to_string(elite));
		if(overflow_elite)
			throw std::overflow_error("Overflow elite value " + std::to_string(elite));
	}
};

/*
template <typename GeneType, typename FitnessType>
class SimpleGenetic: public BaseGenetic<GeneType, FitnessType>
{
	using Individual = BaseIndividual<GeneType, FitnessType>;
	using Population = std::vector<Individual>;

	using Generator = std::function<Individual(std::mt19937&)>;
	using Comparator = std::function<bool(const Individual&, const Individual&)>;
	using Evaluate = std::function<FitnessType(Individual&)>;
	using StopCond = std::function<bool(Individual&)>;

	using Selection = std::function<void(typename Population::iterator, typename Population::iterator, Comparator, std::mt19937&)>;
	using Crossover = std::function<void(typename Population::iterator, typename Population::iterator, std::mt19937&)>;
	using Mutation = std::function<void(typename Population::iterator, typename Population::iterator, std::mt19937&)>;

	public:

	bool output;

	SimpleGenetic(Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Crossover mate, Mutation mut):
		BaseGenetic<GeneType, FitnessType>(gen, comp, eval, stop, sel, mate, mut)
	{	
	}

	SimpleGenetic(int max_gen, int pop_size, int elite, Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Crossover mate, Mutation mut):
		BaseGenetic<GeneType, FitnessType>(max_gen, pop_size, elite, gen, comp, eval, stop, sel, mate, mut)
	{	
	}

	Individual operator()(void) override
	{
		Population population(this->pop_size);
		std::generate(population.begin(), population.end(), [this](){ return this->generator(this->engine); });
		int evaluated_counter = compute_fitness(population);

		Individual best = this->get_best(population, this->comparator);
		Individual worst = this->get_worst(population, this->comparator);

		auto ch = '\t';

		if (output)
		{
			std::cout << "iter" << ch << "evals" << ch << "low-fit" << ch << "high-fit" << std::endl;
			std::cout << 0 << ch << evaluated_counter << ch << worst.fitness << ch << best.fitness << std::endl;
		}

		if(this->stop_cond(best))
		{
			//std::cout << "Success!" << std::endl;
			return best;
		}

		for(iter_until_convergence = 1; iter_until_convergence <= this->max_gen; iter_until_convergence++)
		{
			evaluated_counter = one_iter(population);

			best = this->get_best(population, this->comparator);
			worst = this->get_worst(population, this->comparator);

			if (output)
				std::cout << iter_until_convergence << ch << evaluated_counter << ch << worst.fitness << ch << best.fitness << ch << std::endl;

			if(this->stop_cond(best))
			{
				//std::cout << "Success!" << std::endl;
				return best;
			}
		}	

		return best;
	}

	//private:

	int one_iter(Population &pop)
	{
		this->sel(pop, this->comparator, this->engine);
		this->mate(pop.begin(), pop.end(), this->engine);
		this->mut(pop.begin(), pop.end(), this->engine);

		return compute_fitness(pop);
	}

	int compute_fitness(Population &pop)
	{
		int evaluated_counter = 0;

		for(Individual &i: pop)
		{
			if(!i.is_valid())
			{
				i.compute_fitness(this->evaluate);

				evaluated_counter++;
			}
		}
		return evaluated_counter;
	}
};
*/

/*
template <typename GeneType, typename FitnessType>
std::ostream& operator<<(std::ostream &stream, const BaseIndividual<GeneType, FitnessType> &it)
{
	for(auto i = it.genes.begin(); i != it.genes.end(); i++)
	{
		stream << *i;
		if (i + 1 != it.genes.end())
			stream << ' ';
	}
	stream << ' ' << it.fitness;

	return stream;
}
*/

template <typename GeneType, typename FitnessType>
class AdvancedGenetic: public BaseGenetic<GeneType, FitnessType>
{	
	using Individual = BaseIndividual<GeneType, FitnessType>;
	using Population = std::vector<Individual>;
	using Generator = std::function<Individual(std::mt19937&)>;
	using Comparator = std::function<bool(const Individual&, const Individual&)>;
	using Evaluate = std::function<FitnessType(Individual&)>;
	using StopCond = std::function<bool(Individual&)>;
	using Selection = std::function<void(typename Population::iterator, typename Population::iterator, Comparator, std::mt19937&)>;
	using Crossover = std::function<void(typename Population::iterator, typename Population::iterator, std::mt19937&)>;
	using Mutation =  std::function<void(typename Population::iterator, typename Population::iterator, std::mt19937&)>;

	std::atomic<int> tasks_count;
	std::atomic<int> evaluated_counter;
	std::mutex mutex;
	std::condition_variable cond_var;
	int threads_count;
	thread_pool pool;

	public:

	AdvancedGenetic(Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Crossover mate, Mutation mut):
		BaseGenetic<GeneType, FitnessType>(gen, comp, eval, stop, sel, mate, mut),
		threads_count(std::thread::hardware_concurrency()),
		pool(threads_count)
	{	}

	AdvancedGenetic(int max_gen, int pop_size, int elite, Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Crossover mate, Mutation mut):
		BaseGenetic<GeneType, FitnessType>(max_gen, pop_size, elite, gen, comp, eval, stop, sel, mate, mut),
		threads_count(std::thread::hardware_concurrency()),
		pool(threads_count)
	{	}

	/*
	Individual operator()(void)
	{
		Population population(this->pop_size);
		std::generate(population.begin(), population.end(), [this](){ return this->generator(this->engine); });

		for(iter_until_convergence = 0; iter_until_convergence < this->max_gen; iter_until_convergence++)
		{
			this->sel(population, this->comparator, this->engine);
			mate_parallel(population);
			mutation_parallel(population);

			int eval_count = compute_fitness(population);

			Individual best = this->get_best(population, this->comparator);
			Individual worst = this->get_worst(population, this->comparator);

			if (output)
			{
				std::cout << iter_until_convergence << '\t' << eval_count << '\t' << worst.fitness << '\t' << best.fitness << std::endl;
			}

			if (this->stop_cond(best))
			{
				std::cout << "Success" << std::endl;
				return best;
			}
		}

		return this->get_best(population, this->comparator);
	}
	*/

	/*
	void init_pop(Population &pop)
	{
		tasks_count = threads_count;
		int h = this->pop_size / threads_count;
		int r = this->pop_size % threads_count;
		int d = 0;

		for(int i = 0; i < threads_count; i++)
		{
			int t = i < r ? 1 : 0;
			auto begin = pop.begin() + d;
			auto end = pop.begin() + d + t + h;

			pool.push([this, begin, end](){
				std::generate(begin, end, this->generator);
				if (!(--tasks_count))
				{
					cond_var.notify_one();
				}
			});
		}

		std::unique_lock<std::mutex> locker(mutex);
		cond_var.wait(locker, [this](){ return !tasks_count; });
	}
	*/

	int evaluate_pop(typename Population::iterator begin, typename Population::iterator end) override
	{
		evaluated_counter = 0;
		tasks_count = threads_count;

		int pop_size = std::distance(begin, end);

		int h = pop_size / threads_count;
		int r = pop_size % threads_count;
		int chunk = 0;

		for(int i = 0; i < threads_count; i++)
		{
			int t = i < r ? 1 : 0;
			auto local_begin = begin + chunk;
			auto local_end = begin + chunk + t + h;

			pool.push([this, local_begin, local_end](){
				std::for_each(local_begin, local_end, [this](Individual &i){
						if (!i.is_valid())
						{
							++evaluated_counter;
							i.evaluate(this->evaluate);
						}
				});
				if (!(--tasks_count))
				{
					cond_var.notify_one();
				}
			});

			chunk += t + h;
		}

		std::unique_lock<std::mutex> locker(mutex);
		cond_var.wait(locker, [this](){ return !tasks_count; });

		return evaluated_counter;
	}

	void crossover_operator(typename Population::iterator begin, typename Population::iterator end) override
	{
		tasks_count = threads_count;

		int pop_size = std::distance(begin, end);

		int h = pop_size / threads_count;
		int r = pop_size % threads_count;
		int chunk = 0;

		for(int i = 0; i < threads_count; i++)
		{
			int t = i < r ? 1 : 0;
			auto local_begin = begin + chunk;
			auto local_end = begin + chunk + t + h;

			pool.push([this, local_begin, local_end, i](){
				this->mate(local_begin, local_end, this->mt[i]);
				if (!(--tasks_count))
				{
					cond_var.notify_one();
				}
			});

			chunk += t + h;
		}

		std::unique_lock<std::mutex> locker(mutex);
		cond_var.wait(locker, [this](){ return !tasks_count; });
	}
	
	void mutation_operator(typename Population::iterator begin, typename Population::iterator end) override
	{
		tasks_count = threads_count;

		int pop_size = std::distance(begin, end);

		int h = pop_size / threads_count;
		int r = pop_size % threads_count;
		int chunk = 0;

		for(int i = 0; i < threads_count; i++)
		{
			int t = i < r ? 1 : 0;
			auto local_begin = begin + chunk;
			auto local_end = begin + chunk + t + h;

			pool.push([this, local_begin, local_end, i](){
				this->mutate(local_begin, local_end, this->mt[i]);
				if (!(--tasks_count))
				{
					cond_var.notify_one();
				}
			});

			chunk += t + h;
		}

		std::unique_lock<std::mutex> locker(mutex);
		cond_var.wait(locker, [this](){ return !tasks_count; });
	}

	void init_mt(void) override
	{
		std::random_device rd;

		for(int i = 0; i < threads_count; i++)
		{
			this->mt.push_back(std::mt19937(rd()));
		}
	}

	/*
	void print(Population &pop)
	{
		for(auto &i: pop)
		{
			for(auto &g: i.genes)
			{
				std::cout << g << ' ';
			}
			std::cout << ' ' << i.fitness << std::endl;
		}
	}
	*/
};

template <typename GeneType, typename FitnessType>
auto make_genetic()
{
	return 0;
}

} // namespace ga

#endif

/*
#include <cmath>
#include <chrono>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

const int lim_i = 2e3;
const int lim_j = 1e3;
const int lim_create = 1e8;

struct A
{
	typedef std::function<void(int)> F1;
	typedef std::function<void(int)> F2;
	typedef std::function<void(int)> F3;

	F1 f1;
	F2 f2;
	F3 f3;

	A(F1 f1, F2 f2, F3 f3): f1(f1), f2(f2), f3(f3)
	{	}

	void operator()(void)
	{
		for(int i = 0; i < lim_i; i++)
		{
			for(int j = 0; j < lim_j; j++)
			{
				f1(j);
				f2(j);
				f3(j);
			}
		}
	}
};

template <typename F1, typename F2, typename F3>
struct B
{
	F1 f1;
	F2 f2;
	F3 f3;

	B(F1 f1, F2 f2, F3 f3): f1(f1), f2(f2), f3(f3)
	{	}

	void operator()(void)
	{
		for(int i = 0; i < lim_i; i++)
		{
			for(int j = 0; j < lim_j; j++)
			{
				f1(j);
				f2(j);
				f3(j);
			}
		}
	}
};

template <typename F1, typename F2, typename F3>
auto make_B(F1 f1, F2 f2, F3 f3)
{
	return B<F1, F2, F3>(f1, f2, f3);
}

struct f1
{
	f1() = default;
	void operator()(int n)
	{
		int s = 0;
		for(int i = 0; i < n; i++)
		{
			s += std::sqrt(i);
		}
	}
};

struct f2
{
	f2() = default;
	void operator()(int n)
	{
		f1()(n);
	}
};

struct f3
{
	f3() = default;
	void operator()(int n)
	{
		f2()(n);
	}
};

int main(void)
{
	f1 f_1;
	f2 f_2;
	f3 f_3;

	auto t1 = std::chrono::steady_clock::now();
	for(int i = 0; i < lim_create; i++)
		A a(f_3, f_3, f_3);
	auto t2 = std::chrono::steady_clock::now();
	std::chrono::duration<double> res = t2 - t1;
	std::cout << res.count() << " seconds object A was initialized" << std::endl;

	A a(f_3, f_3, f_3);

	t1 = std::chrono::steady_clock::now();
	a();
	t2 = std::chrono::steady_clock::now();
	res = t2 - t1;
	std::cout << res.count() << " seconds operator A::operator() was using" << std::endl;

	t1 = std::chrono::steady_clock::now();
	for(int i = 0; i < lim_create; i++)
	{
		auto b = make_B(f_3, f_3, f_3);
	}
	t2 = std::chrono::steady_clock::now();
	res = t2 - t1;
	std::cout << res.count() << " seconds object B was initialized using make_B" << std::endl;

	auto b = make_B(f_3, f_3, f_3);

	t1 = std::chrono::steady_clock::now();
	b();
	t2 = std::chrono::steady_clock::now();
	res = t2 - t1;
	std::cout << res.count() << " seconds operator B::operator() was using" << std::endl;

	t1 = std::chrono::steady_clock::now();
	for(int i = 0; i < lim_create; i++)
	{
		B<f1, f2, f3> b(f_1, f_2, f_3);
	}
	t2 = std::chrono::steady_clock::now();
	res = t2 - t1;
	std::cout << res.count() << " seconds object B was initialized" << std::endl;

	B<f1, f2, f3> b2(f_1, f_2, f_3);

	t1 = std::chrono::steady_clock::now();
	b2();
	t2 = std::chrono::steady_clock::now();
	res = t2 - t1;
	std::cout << res.count() << " seconds operator B::operator() was using" << std::endl;

	return 0;
}

*/
