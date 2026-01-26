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

	void compute_fitness(std::function<FitnessType(BaseIndividual<GeneType, FitnessType> &i)> fit_func)
	{
		fitness = fit_func(*this);
		valid = true;
	}

	bool is_valid(void)
	{
		return valid;
	}
};

template <typename GeneType, typename FitnessType>
struct BaseGenetic
{
	using Individual = BaseIndividual<GeneType, FitnessType>;
	using Population = std::vector<Individual>;
	using Generator = std::function<Individual(void)>;
	using Comparator = std::function<bool(const Individual&, const Individual&)>;
	using Evaluate = std::function<FitnessType(Individual&)>;
	using StopCond = std::function<bool(Individual&)>;
	using Selection = std::function<void(Population&, Comparator, std::mt19937&)>;
	using Crossover = std::function<void(typename Population::iterator, typename Population::iterator, std::mt19937&)>;
	using Mutation = std::function<void(typename Population::iterator, typename Population::iterator, std::mt19937&)>;

	std::random_device rd;
	std::mt19937 engine;

	int max_gen;
	int pop_size;
	int elite;

	Generator generator;
	Comparator comparator;
	Evaluate evaluate;
	StopCond stopcond;

	Selection sel;
	Crossover mate;
	Mutation mut;

	public:

	BaseGenetic(Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Crossover mate, Mutation mut):
		max_gen(50),
		pop_size(100),
		elite(10),
		generator(gen),
		comparator(comp),
		evaluate(eval),
		stopcond(stop),
		sel(sel),
		mate(mate),
		mut(mut),
		engine(rd())
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
		stopcond(stop),
		sel(sel),
		mate(mate),
		mut(mut),
		engine(rd())
	{
		config();
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
			throw std::underflow_error("Overflow elite value " + std::to_string(elite));
	}

	static Individual get_best(Population &pop, Comparator comp)
	{
		auto best = std::min_element(pop.begin(), pop.end(), comp);
		return *best;
	}

	static Individual get_worst(Population &pop, Comparator comp)
	{
		auto worst = std::max_element(pop.begin(), pop.end(), comp);
		return *worst;
	}
};

template <typename GeneType, typename FitnessType>
class SimpleGenetic: private BaseGenetic<GeneType, FitnessType>
{
	using Individual = BaseIndividual<GeneType, FitnessType>;
	using Population = std::vector<Individual>;
	using Generator = std::function<Individual(void)>;
	using Comparator = std::function<bool(const Individual&, const Individual&)>;
	using Evaluate = std::function<FitnessType(Individual&)>;
	using StopCond = std::function<bool(Individual&)>;
	using Selection = std::function<void(Population&, Comparator, std::mt19937&)>;
	using Crossover = std::function<void(typename Population::iterator, typename Population::iterator, std::mt19937&)>;
	using Mutation = std::function<void(typename Population::iterator, typename Population::iterator, std::mt19937&)>;

	Population population;

	public:

	bool output;

	SimpleGenetic(Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Crossover mate, Mutation mut):
		BaseGenetic<GeneType, FitnessType>(gen, comp, eval, stop, sel, mate, mut),
		population(this->pop_size),
		output(true)
	{	
	}

	SimpleGenetic(int max_gen, int pop_size, int elite, Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Crossover mate, Mutation mut):
		BaseGenetic<GeneType, FitnessType>(max_gen, pop_size, elite, gen, comp, eval, stop, sel, mate, mut),
		population(this->pop_size),
		output(true)
	{	
	}

	Individual operator()(void)
	{
		std::generate(population.begin(), population.end(), this->generator);
		int evaluated_counter = compute_fitness(population);

		Individual best = this->get_best(population, this->comparator);
		Individual worst = this->get_worst(population, this->comparator);

		auto ch = '\t';

		if (output)
		{
			std::cout << "iter" << ch << "evals" << ch << "low-fit" << ch << "high-fit" << std::endl;
			std::cout << 0 << ch << evaluated_counter << ch << worst.fitness << ch << best.fitness << std::endl;
		}

		if(this->stopcond(best))
		{
			//std::cout << "Success!" << std::endl;
			return best;
		}

		for(int iter = 1; iter <= this->max_gen; iter++)
		{
			evaluated_counter = one_iter(population);

			best = this->get_best(population, this->comparator);
			worst = this->get_worst(population, this->comparator);

			if (output)
				std::cout << iter << ch << evaluated_counter << ch << worst.fitness << ch << best.fitness << ch << std::endl;

			if(this->stopcond(best))
			{
				//std::cout << "Success!" << std::endl;
				return best;
			}
		}	

		return best;
	}

	private:

	int one_iter(Population &pop)
	{
		this->sel(pop, this->comparator, this->engine);
		this->mate(pop.begin(), pop.end(), this->engine);
		this->mut(pop.begin(), pop.end(), this->engine);

		return compute_fitness(population);
	}

	int compute_fitness(Population &population)
	{
		int evaluated_counter = 0;

		for(Individual &i: population)
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

template <typename GeneType, typename FitnessType>
class AdvancedGenetic: BaseGenetic<GeneType, FitnessType>
{	
	using Individual = BaseIndividual<GeneType, FitnessType>;
	using Population = std::vector<Individual>;
	using Generator = std::function<Individual(void)>;
	using Comparator = std::function<bool(const Individual&, const Individual&)>;
	using Evaluate = std::function<FitnessType(Individual&)>;
	using StopCond = std::function<bool(Individual&)>;
	using Selection = std::function<void(Population&, Comparator, std::mt19937&)>;
	using Crossover = std::function<void(typename Population::iterator, typename Population::iterator, std::mt19937&)>;
	using Mutation =  std::function<void(typename Population::iterator, typename Population::iterator, std::mt19937&)>;

	std::atomic<int> tasks_count;
	std::atomic<int> evaluated_counter;
	std::mutex mutex;
	std::condition_variable cond_var;
	int threads_count;
	thread_pool pool;

	public:

	bool output;

	AdvancedGenetic(Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Crossover mate, Mutation mut):
		BaseGenetic<GeneType, FitnessType>(gen, comp, eval, stop, sel, mate, mut),
		threads_count(std::thread::hardware_concurrency()),
		pool(threads_count),
		output(true)
	{	}

	AdvancedGenetic(int max_gen, int pop_size, int elite, Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Crossover mate, Mutation mut):
		BaseGenetic<GeneType, FitnessType>(max_gen, pop_size, elite, gen, comp, eval, stop, sel, mate, mut),
		threads_count(std::thread::hardware_concurrency()),
		pool(threads_count),
		output(true)
	{	}

	Individual operator()(void)
	{
		Population population(this->pop_size);
		std::generate(population.begin(), population.end(), this->generator);

		for(int iter = 0; iter < this->max_gen; iter++)
		{
			this->sel(population, this->comparator, this->engine);
			mate_parallel(population);
			mutation_parallel(population);

			int eval_count = compute_fitness(population);

			Individual best = this->get_best(population, this->comparator);
			Individual worst = this->get_worst(population, this->comparator);

			if (output)
			{
				std::cout << iter << '\t' << eval_count << '\t' << worst.fitness << '\t' << best.fitness << std::endl;
			}

			if (this->stopcond(best))
			{
				std::cout << "Success" << std::endl;
				return best;
			}
		}

		return this->get_best(population, this->comparator);
	}

	private:

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

	int compute_fitness(Population &pop)
	{
		evaluated_counter = 0;
		tasks_count = threads_count;

		int h = this->pop_size / threads_count;
		int r = this->pop_size % threads_count;
		int chunk = 0;

		for(int i = 0; i < threads_count; i++)
		{
			int t = i < r ? 1 : 0;
			auto begin = pop.begin() + chunk;
			auto end = pop.begin() + chunk + t + h;

			pool.push([this, begin, end](){
				std::for_each(begin, end, [this](Individual &i){
						if (!i.valid)
						{
							++evaluated_counter;
							i.compute_fitness(this->evaluate);
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

	void mutation_parallel(Population &pop)
	{
		tasks_count = threads_count;
		int h = this->pop_size / threads_count;
		int r = this->pop_size % threads_count;
		int chunk = 0;

		for(int i = 0; i < threads_count; i++)
		{
			int t = i < r ? 1 : 0;
			auto begin = pop.begin() + chunk;
			auto end = pop.begin() + chunk + t + h;

			pool.push([this, begin, end](){
				this->mut(begin, end, this->engine);
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

	void mate_parallel(Population &pop)
	{
		tasks_count = threads_count;
		int h = this->pop_size / threads_count;
		int r = this->pop_size % threads_count;
		int chunk = 0;

		for(int i = 0; i < threads_count; i++)
		{
			int t = i < r ? 1 : 0;
			auto begin = pop.begin() + chunk;
			auto end = pop.begin() + chunk + t + h;

			pool.push([this, begin, end](){
				this->mate(begin, end, this->engine);
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
};

} // namespace ga

#endif
