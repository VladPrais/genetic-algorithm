#ifndef GA_H
#define GA_H

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include "thread_pool.h"

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

	/*
	void set_genes(GeneType &new_genes)
	{
		genes = new_genes;
		valid = false;
	}

	void set_fitness(FitnessType new_fitness)
	{
		fitness = new_fitness;
		valid = true;
	}

	const GeneType& get_genes(void) const
	{
		return genes;
	}

	const FitnessType& get_fitness(void) const
	{
		return fitness;
	}
	*/

	bool is_valid(void) const
	{
		return valid;
	}
};

template <typename FitnessType>
struct statistics
{
	std::vector<FitnessType> bests;
	std::vector<FitnessType> worsts;
	size_t generations;
};

template <typename T, typename Individual, typename Generate, typename Compare, typename Evaluate, typename StopCond>
struct AbstractGenetic
{
	typedef std::vector<Individual> Population;
	typedef typename Population::iterator Iterator;

	Generate generate;
	Compare compare;
	Evaluate evaluate;
	StopCond stopcond;

	std::mt19937 random_generator;
	std::uniform_real_distribution<double> pb_dist;

	AbstractGenetic(Generate gen, Compare comp, Evaluate eval, StopCond stop):
		generate(gen),
		compare(comp),
		evaluate(eval),
		stopcond(stop),
		pb_dist(0.0, 1.0),
		random_generator(std::random_device{}())
	{	}

	template <typename Crossover>
	void crossing(Population& population, Crossover cross, double cxpb)
	{	}

	template <typename Mutation>
	void mutating(Population& population, Mutation mutate, double mtpb)
	{	}

	int evaluating(Population& population)
	{
		return 0;
	}

	double get_random_value(void)
	{
		return this->pb_dist(this->random_generator);
	}

	template <typename Population>
	Individual get_best(Population& population)
	{
		Individual best = *std::min_element(population.begin(), population.end(), this->compare);
		return best;
	}

	template <typename Population>
	Individual get_worst(Population& population)
	{
		Individual worst = *std::max_element(population.begin(), population.end(), this->compare);
		return worst;
	}

	auto time(void)
	{
		return std::chrono::steady_clock::now();
	}

	template <typename TimeType>
	double tresult(TimeType t1, TimeType t2)
	{
		std::chrono::duration<double> t = t2 - t1;
		return t.count();
	}

	template <typename FitnessType>
	static void print_stat(size_t gen, size_t eval_count, const FitnessType& bad, const FitnessType& good)
	{
		char ch = '\t';
		std::cout << gen << ch << eval_count << ch << bad << ch << good << std::endl;
	}

	template <typename Selection, typename Crossover, typename Mutation>
	Individual operator()(size_t gen_limit, size_t pop_size, size_t elite_size, double cxpb, double mtpb, Selection selecting, Crossover cross, Mutation mutate)
	{
		auto time_start = time();

		Population population(pop_size), offspring(pop_size);
		std::generate(population.begin(), population.end(), [this](){ return this->generate(); });

		int eval_count = static_cast<T*>(this) -> evaluating(population);

		Individual best = get_best(population);
		Individual worst = get_worst(population);

		print_stat(0, eval_count, worst.fitness, best.fitness);

		if(stopcond(best))
		{
			goto finish;
		}

		for(int i = 1; i <= gen_limit; i++)
		{
			std::nth_element(population.begin(), population.begin() + elite_size, population.end(), this->compare);
			std::copy(population.begin(), population.begin() + elite_size, offspring.begin());

			selecting(population.begin(), population.end(), offspring.begin() + elite_size, offspring.end());
			static_cast<T*>(this) -> crossing(offspring.begin() + elite_size, offspring.end(), cross, cxpb);
			static_cast<T*>(this) -> mutating(offspring.begin() + elite_size, offspring.end(), mutate, mtpb);

			std::swap(population, offspring);

			eval_count = static_cast<T*>(this) -> evaluating(population);

			best = get_best(population);
			worst = get_worst(population);

			print_stat(i, eval_count, worst.fitness, best.fitness);

			if(stopcond(best))
			{
				break;
			}
		}

		finish:

		auto time_end = time();
		std::cout << tresult(time_start, time_end) << " seconds." << std::endl;

		return best;
	}
};

template <typename T, typename Individual, typename Generate, typename Compare, typename Evaluate, typename StopCond>
struct AbstractCommonGenetic: AbstractGenetic<T, Individual, Generate, Compare, Evaluate, StopCond>
{
	typedef typename AbstractGenetic<T, Individual, Generate, Compare, Evaluate, StopCond>::Population Population;

	AbstractCommonGenetic(Generate gen, Compare comp, Evaluate eval, StopCond stop):
		AbstractGenetic<T, Individual, Generate, Compare, Evaluate, StopCond>(gen, comp, eval, stop)
	{	}

	template <typename Iterator, typename Crossover>
	void crossing(Iterator begin, Iterator end, Crossover cross, double cxpb)
	{
		size_t pop_size = std::distance(begin, end);

		for(int i = 0; i < pop_size - 1; i += 2)
		{
			if(this->get_random_value() < cxpb)
			{
				cross(*begin, *(begin + 1));
				begin->valid = false;
				(begin + 1)->valid = false;
				begin++;
			}
		}
	}

	template <typename Iterator, typename Mutation>
	void mutating(Iterator begin, Iterator end, Mutation mutate, double mtpb)
	{
		size_t pop_size = std::distance(begin, end);

		for(int i = 0; i < pop_size; i++)
		{
			if(this->get_random_value() < mtpb)
			{
				mutate(*begin);
				begin->valid = false;
				begin++;
			}
		}
	}

	int evaluating(Population& population)
	{
		int eval_count = 0;

		std::for_each(population.begin(), population.end(), [&eval_count, this](Individual& i){
			if(!i.is_valid())
			{
				i.fitness = this->evaluate(i);
				i.valid = true;
				eval_count++;
			} 
		});

		return eval_count;
	}

};

template <typename Individual, typename Generate, typename Compare, typename Evaluate, typename StopCond>
struct CommonGenetic: AbstractCommonGenetic<CommonGenetic<Individual, Generate, Compare, Evaluate, StopCond>, Individual, Generate, Compare, Evaluate, StopCond>
{
	CommonGenetic(Generate g, Compare c, Evaluate e, StopCond s):
		AbstractCommonGenetic<CommonGenetic<Individual, Generate, Compare, Evaluate, StopCond>, Individual, Generate, Compare, Evaluate, StopCond>(g, c, e, s)
	{	}
};

template <typename Individual, typename Generate, typename Compare, typename Evaluate, typename StopCond>
CommonGenetic<Individual, Generate, Compare, Evaluate, StopCond> make_common(Generate g, Compare c, Evaluate e, StopCond s)
{
	return CommonGenetic<Individual, Generate, Compare, Evaluate, StopCond>(g, c, e, s);
}

template <typename T, typename Individual, typename Generate, typename Compare, typename Evaluate, typename StopCond>
struct AbstractAdvancedGenetic: AbstractCommonGenetic<T, Individual, Generate, Compare, Evaluate, StopCond>
{
	typedef typename AbstractGenetic<T, Individual, Generate, Compare, Evaluate, StopCond>::Population Population;

	std::mutex mtx;
	std::condition_variable cond_var;
	size_t threads_count;
	thread_pool pool;

	AbstractAdvancedGenetic(Generate g, Compare c, Evaluate e, StopCond s):
		AbstractCommonGenetic<T, Individual, Generate, Compare, Evaluate, StopCond>(g, c, e, s),
		threads_count(std::thread::hardware_concurrency()),
		pool(threads_count)
	{	}

	size_t evaluating(Population& population)
	{
		std::atomic<size_t> eval_count {0};
		std::atomic<size_t> tasks_count {threads_count};

		int pop_size = population.size();

		int h = pop_size / threads_count;
		int r = pop_size % threads_count;
		int chunk = 0;

		for(int i = 0; i < threads_count; i++)
		{
			int t = i < r ? 1 : 0;
			auto local_begin = population.begin() + chunk;
			auto local_end = population.begin() + chunk + t + h;

			auto task = [this, local_begin, local_end, &eval_count, &tasks_count]()
			{
				size_t local_eval_count = 0;

				std::for_each(local_begin, local_end, [this, &local_eval_count](Individual& i)
						{
							if(!i.is_valid())
							{
								i.fitness = this->evaluate(i);
								i.valid = true;
								local_eval_count++;
							}
						});
				eval_count += local_eval_count;
				if(!(--tasks_count))
				{
					std::unique_lock<std::mutex> locker(this->mtx);
					this->cond_var.notify_one();
				}
			};
			pool.push(task);

			chunk += t + h;
		}

		std::unique_lock<std::mutex> locker(mtx);
		cond_var.wait(locker, [&tasks_count](){ return !tasks_count; });

		return static_cast<size_t>(eval_count);
	}

};

template <typename Individual, typename Generate, typename Compare, typename Evaluate, typename StopCond>
struct AdvancedGenetic: AbstractAdvancedGenetic<AdvancedGenetic<Individual, Generate, Compare, Evaluate, StopCond>, Individual, Generate, Compare, Evaluate, StopCond>
{
	AdvancedGenetic(Generate g, Compare c, Evaluate e, StopCond s):
		AbstractAdvancedGenetic<AdvancedGenetic<Individual, Generate, Compare, Evaluate, StopCond>, Individual, Generate, Compare, Evaluate, StopCond>(g, c, e, s)
	{	}
};

template <typename Individual, typename Generate, typename Compare, typename Evaluate, typename StopCond>
AdvancedGenetic<Individual, Generate, Compare, Evaluate, StopCond> make_advanced(Generate g, Compare c, Evaluate e, StopCond s)
{
	return AdvancedGenetic<Individual, Generate, Compare, Evaluate, StopCond>(g, c, e, s);
}

/*
template <typename GeneType, typename FitnessType>
struct ParallelGenetic: BaseGenetic<GeneType, FitnessType>
{	
	typedef BaseIndividual<GeneType, FitnessType> Individual;
	typedef std::vector<Individual> Population;
	typedef typename Population::iterator Iterator;

	typedef std::function<Individual(std::mt19937&)> Generator;
	typedef std::function<bool(const Individual&, const Individual&)> Comparator;
	typedef std::function<FitnessType(Individual&)> Evaluate;
	typedef std::function<bool(Individual&)> StopCond ;

	typedef std::function<void(Iterator, Iterator, Comparator, std::mt19937&)> Selection;
	typedef std::function<void(Iterator, Iterator, std::mt19937&)> Crossover;
	typedef std::function<void(Iterator, Iterator, std::mt19937&)> Mutation;

	std::atomic<int> tasks_count;
	std::atomic<int> evaluated_count;
	std::mutex mutex;
	std::condition_variable cond_var;
	int threads_count;
	thread_pool pool;

	ParallelGenetic(Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Crossover mate, Mutation mut, int max_gen=200, int pop_size=1000, int elite=50):
		BaseGenetic<GeneType, FitnessType>(gen, comp, eval, stop, sel, mate, mut, max_gen, pop_size, elite),
		threads_count(std::thread::hardware_concurrency()),
		pool(threads_count)
	{
		std::cout << threads_count << std::endl;
	}

	int genetic_init(Iterator begin, Iterator end) override
	{
		std::random_device rd;

		for(int i = 0; i < threads_count; i++)
		{
			this->mt.push_back(std::mt19937(rd()));
		}

		evaluated_count = 0;
		tasks_count = threads_count;

		int h = this->pop_size / threads_count;
		int r = this->pop_size % threads_count;
		int chunk = 0;

		for(int i = 0; i < threads_count; i++)
		{
			int t = i < r ? 1 : 0;
			Iterator local_begin = begin + chunk;
			Iterator local_end = begin + chunk + t + h;

			pool.push([this, local_begin, local_end, i](){
				std::generate(local_begin, local_end, [this, i](){ return this->generator(this->mt[i]); });
				evaluated_count += this->evaluate_slice(local_begin, local_end);
				if (!(--tasks_count))
				{
					cond_var.notify_one();
				}
			});

			chunk += t + h;
		}

		std::unique_lock<std::mutex> locker(mutex);
		cond_var.wait(locker, [this](){ return !tasks_count; });

		return evaluated_count;
	}

	int genetic_step(Iterator begin, Iterator end) override
	{
		this->select(begin, end, this->comparator, this->mt[0]);

		tasks_count = threads_count;
		evaluated_count = 0;

		int h = this->pop_size / threads_count;
		int r = this->pop_size % threads_count;
		int chunk = 0;

		for(int i = 0; i < threads_count; i++)
		{
			int t = i < r ? 1 : 0;

			Iterator local_begin = begin + chunk;
			Iterator local_end = begin + chunk + h + t;

			pool.push([this, local_begin, local_end, i](){
				this->mate(local_begin, local_end, this->mt[i]);
				this->mutate(local_begin, local_end, this->mt[i]);
				evaluated_count += this->evaluate_slice(local_begin, local_end);
				if (!(--tasks_count))
				{
					cond_var.notify_one();
				}
			});

			chunk += h + t;
		}

		std::unique_lock<std::mutex> locker(mutex);
		cond_var.wait(locker, [this](){ return !tasks_count; });

		return evaluated_count;
	}

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
	*/

	/*
	void crossover_operator(typename Population::iterator begin, typename Population::iterator end)
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
	
	void mutation_operator(typename Population::iterator begin, typename Population::iterator end)
	{
		tasks_coun = threads_count;

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
};
*/

/*
 * Selection operator
 */

/*
template <typename Individual>
struct Tournament
{
	std::mt19937 engine;
	
	Toutnament(void)

	void operator()(typename std::vector<Individual>::iterator begin, typename std::vector<Individual>::iterator end , Comparator comparator, std::mt19937 &engine)
	{
		int pop_size = std::distance(begin, end);
		std::vector<Individual> aspirants(pop_size);

		aut get_best = [&](){
			std::vector<Individual> rnd = sel_random<Individual>(begin, end, tourn_size, engine);
			return *std::min_element(rnd.begin(), rnd.end(), comparator);
		};

		std::generate(aspirants.begin(), aspirants.end(), get_best);

		std::copy(aspirants.begin(), aspirants.end(), begin);
	}
};
*/

template <typename Iterator, typename Compare>
void sel_tournament(Iterator begin_pop, Iterator end_pop, Iterator begin_off, Iterator end_off, Compare compare, std::mt19937& generator, size_t tourn_size = 3)
{
	size_t pop_size = std::distance(begin_pop, end_pop), off_size = std::distance(begin_off, end_off);
	std::uniform_int_distribution<int> dist(0, pop_size - 1);
	Iterator best, choice;

	for(size_t i = 0; i < off_size; i++)
	{
		best = begin_pop + dist(generator);
		for(size_t j = 1; j < tourn_size; j++)
		{
			choice = begin_pop + dist(generator);
			best = compare(*best, *choice) ? best : choice;
		}
		*begin_off = *best;
		begin_off++;
	}
}

/*
 * Crossover operator
 */

template <typename Individual, typename Generator>
void cross_one_point(Individual &lhs, Individual &rhs, Generator& generator)
{
	int size = std::min(lhs.genes.size(), rhs.genes.size());
	std::uniform_int_distribution<int> dist(1, size - 2);
	int cross_point = dist(generator);
	//auto g1 = lhs.get_genes(), g2 = rhs.get_genes();

	for(int i = 0; i < cross_point; i++)
	{
		std::swap(lhs.genes[i], rhs.genes[i]);
	}
}

/*
 * Mutation operator
 */

template <typename Individual, typename Generator>
void mut_bit_not(Individual& object, Generator& generator)
{
	size_t len_genes = object.genes.size();
	std::uniform_int_distribution<int> dist(0, len_genes - 1);
	size_t i = dist(generator);
	object.genes[i] = !object.genes[i];
}

template <typename Individual, typename Generator>
void mut_bit_not_uniform(Individual& object, Generator& generator, double mgpb = 0.01)
{
	size_t len_genes = object.genes.size();
	std::uniform_real_distribution<double> pb(0.0, 1.0);

	for(size_t i = 0; i < len_genes; i++)
	{
		if(pb(generator) < mgpb)
		{
			object.genes[i] = !object.genes[i];
		}
	}
}

template <typename Individual, typename Generator>
void mut_normal(Individual& object, Generator& generator, double mu = 0.0, double sigma = 1.0)
{
	size_t len_genes = object.genes.size();
	std::uniform_int_distribution<int> dist(0, len_genes - 1);
	std::normal_distribution<double> norm(mu, sigma);
	size_t i = dist(generator);
	object.genes[i] += norm(generator);
}

} // namespace ga

#endif
