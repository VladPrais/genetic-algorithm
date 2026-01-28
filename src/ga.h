#ifndef __GA_H__
#define __GA_H__

#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

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

template <typename Individual>
struct GeneticOutput
{
	int generation_count;
	std::vector<Individual> hall_of_fame;
	Individual best_ever;

	GeneticOutput(int gen_count, const Individual &best):
		generation_count(gen_count),
		best_ever(best)
	{	}
};

//template <typename GeneType, typename FitnessType, typename Selection, typename Crossover, typename Mutation>
template <typename GeneType, typename FitnessType>
struct BaseGenetic
{
	typedef BaseIndividual<GeneType, FitnessType> Individual;
	typedef std::vector<Individual> Population;

	typedef std::function<Individual(std::mt19937&)> Generator;
	typedef std::function<bool(const Individual&, const Individual&)> Comparator;
	typedef std::function<FitnessType(Individual&)> Evaluate;
	typedef std::function<bool(Individual&)> StopCond ;

	typedef std::function<void(typename Population::iterator, typename Population::iterator, Comparator, std::mt19937&)> Selection;
	typedef std::function<void(typename Population::iterator, typename Population::iterator, std::mt19937&)> Crossover;
	typedef std::function<void(typename Population::iterator, typename Population::iterator, std::mt19937&)> Mutation;

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

	BaseGenetic(Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Crossover mate, Mutation mut, int max_gen=50, int pop_size=100, int elite=10):
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

	GeneticOutput<Individual> operator()(void)
	{
		init_mt();

		Population population(pop_size);

		init_pop(population.begin(), population.end());

		if (output)
		{
			std::cout << pop_size << " individuals were initialized" << std::endl;
		}

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
			if (output)
				std::cout << "Success!" << std::endl;

			GeneticOutput gen_out(0, best);
			return gen_out;
		}

		Population hall_of_fame(elite);

		for(int iter = 1; iter <= max_gen; iter++)
		{
			/*
			std::nth_element(population.begin(), population.begin() + elite, population.end());
			std::copy(population.begin(), population.begin() + elite, hall_of_fame.begin());
			*/

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
				if (output)
					std::cout << "Success!" << std::endl;

				GeneticOutput gen_out(iter, best);
				return gen_out;
			}
		}

		GeneticOutput gen_out(max_gen, best);
		return gen_out;
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

	virtual void init_pop(typename Population::iterator begin, typename Population::iterator end)
	{
		std::generate(begin, end, [this](){ return generator(mt[0]); });
	}

	virtual void init_mt(void)
	{
		std::random_device rd;
		mt.push_back(std::mt19937(rd()));
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

	void config(void)
	{
		bool underflow_max_gen = max_gen <= 0;
		bool underflow_pop_size = pop_size <= 0;
		bool underflow_elite = elite < 0;
		bool overflow_elite = elite >= pop_size;

		if (underflow_max_gen)
			throw std::underflow_error("Underflow max generations value " + std::to_string(max_gen));
		if (underflow_pop_size)
			throw std::underflow_error("Underflow population size value " + std::to_string(pop_size));
		if (underflow_elite)
			throw std::underflow_error("Underflow elite value " + std::to_string(elite));
		if (overflow_elite)
			throw std::overflow_error("Overflow elite value " + std::to_string(elite));
	}
};

template <typename GeneType, typename FitnessType>
struct AdvancedGenetic: BaseGenetic<GeneType, FitnessType>
{	
	typedef BaseIndividual<GeneType, FitnessType> Individual;
	typedef std::vector<Individual> Population;

	typedef std::function<Individual(std::mt19937&)> Generator;
	typedef std::function<bool(const Individual&, const Individual&)> Comparator;
	typedef std::function<FitnessType(Individual&)> Evaluate;
	typedef std::function<bool(Individual&)> StopCond ;

	typedef std::function<void(typename Population::iterator, typename Population::iterator, Comparator, std::mt19937&)> Selection;
	typedef std::function<void(typename Population::iterator, typename Population::iterator, std::mt19937&)> Crossover;
	typedef std::function<void(typename Population::iterator, typename Population::iterator, std::mt19937&)> Mutation;

	std::atomic<int> tasks_count;
	std::atomic<int> evaluated_counter;
	std::mutex mutex;
	std::condition_variable cond_var;
	int threads_count;
	thread_pool pool;

	public:

	AdvancedGenetic(Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Crossover mate, Mutation mut, int pop_size=1000, int max_gen=200, int elite=50):
		BaseGenetic<GeneType, FitnessType>(gen, comp, eval, stop, sel, mate, mut, pop_size, max_gen, elite),
		threads_count(std::thread::hardware_concurrency()),
		pool(threads_count)
	{	}

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
};

/*
template <typename GeneType, typename FitnessType, typename Generator, typename Comparator, typename Evaluate, typename StopCond, typename Selection, typename Crossover, typename Mutation>
auto make_genetic(int max_gen, int pop_size, int elite, Generator gen, Comparator comp, Evaluate eval, StopCond stop_cond, Selection sel, Crossover mate, Mutation mut)
{
	return BaseGenetic<GeneType, FitnessType, Selection, Crossover, Mutation>(max_gen, pop_size, elite, gen, comp, eval, stop_cond, sel, mate, mut);
}
*/

/*
 * Selection operator
 */

template <typename Individual>
std::vector<Individual> sel_random(typename std::vector<Individual>::iterator begin, typename std::vector<Individual>::iterator end , size_t n, std::mt19937 &engine)
{
	int pop_size = std::distance(begin, end);
	std::vector<Individual> v(n);
	std::uniform_int_distribution<int> dist(0, pop_size - 1);

	std::generate(v.begin(), v.end(), [&dist, &engine, begin](){ return *(begin + dist(engine)); });

	return v;
}

template <typename Individual>
class tournament
{
	typedef std::function<bool(const Individual&, const Individual&)> Comparator;

	int tourn_size;

	public:

	tournament(void): tourn_size(3)
	{	}

	tournament(int tourn_size): tourn_size(tourn_size)
	{	}

	void operator()(typename std::vector<Individual>::iterator begin, typename std::vector<Individual>::iterator end , Comparator comparator, std::mt19937 &engine)
	{
		int pop_size = std::distance(begin, end);
		std::vector<Individual> aspirants(pop_size);

		auto get_best = [&](){
			std::vector<Individual> rnd = sel_random<Individual>(begin, end, tourn_size, engine);
			return *std::min_element(rnd.begin(), rnd.end(), comparator);
		};

		std::generate(aspirants.begin(), aspirants.end(), get_best);

		std::copy(aspirants.begin(), aspirants.end(), begin);
	}
};

/*
 * Crossover operator
 */

template <typename Individual>
class crossover
{
	typedef std::function<void(Individual&, Individual&, std::mt19937&)> Crossover;

	double cxpb;
	Crossover mate_func;
	std::uniform_real_distribution<double> pb;

	public:
	
	crossover(Crossover mate_func):
		cxpb(0.7),
		mate_func(mate_func),
		pb(0.0, 1.0)
	{	
		config();
	}

	crossover(double cxpb, Crossover mate_func):
		cxpb(cxpb),
		mate_func(mate_func),
		pb(0.0, 1.0)
	{	
		config();
	}

	/*
	void operator()(std::vector<Individual> &aspirants, std::mt19937 &engine)
	{
		int pop_size = aspirants.size();

		if(pop_size % 2 != 0)
			throw std::logic_error("Population size have to be even");

		for(int i = 0; i < pop_size; i += 2)
		{
			double p = pb(engine);

			if(p < cxpb)
			{
				mate_func(aspirants[i], aspirants[i + 1], engine);

				aspirants[i].valid = false;
				aspirants[i + 1].valid = false;
			}
		}
	}
	*/
	
	template <typename Iterator>
	void operator()(Iterator begin, Iterator end, std::mt19937 &engine)
	{
		int pop_size = std::distance(begin, end);
		int inc = 0;

		if(pop_size % 2 != 0)
			inc = 1;
//			throw std::logic_error("Population size have to be even");

		for(Iterator i = begin; i + inc != end; i += 2)
		{
			double p = pb(engine);

			if(p < cxpb)
			{
				mate_func(*i, *(i + 1), engine);

				i->valid = false;
				(i + 1)->valid = false;
			}
		}
	}

	private:

	void config(void)
	{
		if(cxpb > 1.0)
			throw std::overflow_error("Overflow crossover probability: " + std::to_string(cxpb));
		if(cxpb < 0.0)
			throw std::underflow_error("Underflow crossover probability: " + std::to_string(cxpb));
	}
};

template <typename Individual>
struct cx_one_point
{
	void operator()(Individual &lhs, Individual &rhs, std::mt19937 &engine)
	{
		int size = std::min(lhs.genes.size(), rhs.genes.size());
		std::uniform_int_distribution<int> dist(1, size - 1);
		int point = dist(engine);

		for(int i = 0; i < point; i++)
		{
			std::swap(lhs.genes[i], rhs.genes[i]);
		}
	}
};

/*
 * Mutation operator
 */

template <typename Individual>
class mutation
{
	typedef std::function<void(Individual&, double, std::mt19937&)> Mutation;
	std::uniform_real_distribution<double> pb;

	double mtpb;
	double mgpb;
	Mutation mut_func;

	public:

	mutation(void):
		mtpb(0.1),
		mgpb(0.05),
		pb(0.0, 1.0)
	{
		config();
	}

	mutation(double mtpb, double mgpb, Mutation mut_func):
		mtpb(mtpb),
		mgpb(mgpb),
		pb(0.0, 1.0),
		mut_func(mut_func)
	{
		config();
	}

	/*
	void operator()(std::vector<Individual> &child, std::mt19937 &engine)
	{
		int pop_size = child.size();

		for(int i = 0; i < pop_size; i++)
		{
			if(pb(engine) < mtpb)
			{
				mut_func(child[i], mgpb, engine);
				child[i].valid = false;
			}
		}
	}
	*/

	void operator()(typename std::vector<Individual>::iterator begin, typename std::vector<Individual>::iterator end, std::mt19937 &engine)
	{
		for(auto i = begin; i != end; i++)
		{
			if(pb(engine) < mtpb)
			{
				mut_func(*i, mgpb, engine);
				i->valid = false;
			}
		}
	}

	private:

	void config(void)
	{
		if(mtpb > 1.0)
			throw std::overflow_error("Overflow mutation probability: " + std::to_string(mtpb));
		if(mgpb > 1.0)
			throw std::overflow_error("Overflow mutation of gene probability: " + std::to_string(mgpb));
		if(mtpb < 0.0)
			throw std::underflow_error("Underflow mutation probability: " + std::to_string(mtpb));
		if(mgpb < 0.0)
			throw std::underflow_error("Underflow mutation of gene probability: " + std::to_string(mgpb));
	}
};

template <typename Individual>
struct mut_gaussian
{
	double mu;
	double sigma;
	std::uniform_real_distribution<double> pb;

	mut_gaussian(void): mu(0.0), sigma(1.0), pb(0.0, 1.0)
	{	}

	mut_gaussian(double mu, double sigma): mu(mu), sigma(sigma), pb(0.0, 1.0)
	{	}

	void operator()(Individual &object, double mgpb, std::mt19937 &engine)
	{
		std::normal_distribution<double> norm(mu, sigma);
		int size = object.genes.size();

		for(int i = 0; i < size; i++)
		{
			if(pb(engine) < mgpb)
				object.genes[i] += norm(engine);
		}
	}
};

template <typename Individual>
struct mut_inverse
{
	std::uniform_real_distribution<double> pb;

	mut_inverse(void): pb(0.0, 1.0)
	{	}

	void operator()(Individual &object, double mgpb, std::mt19937 &engine)
	{
		int size = object.genes.size();

		for(int i = 0; i < size; i++)
		{
			if(pb(engine) < mgpb)
				object.genes[i] = !object.genes[i];
		}
	}
};

} // namespace ga

#endif
