#ifndef __GA_H__
#define __GA_H__

#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "operators_ga.h"

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
class BaseGenetic
{
	using Individual = BaseIndividual<GeneType, FitnessType>;
	using Population = std::vector<Individual>;
	using Generator = std::function<Individual(void)>;
	using Comparator = std::function<bool(const Individual&, const Individual&)>;
	using Evaluate = std::function<FitnessType(Individual&)>;
	using StopCond = std::function<bool(Individual&)>;
	using Selection = std::function<void(Population&, Comparator, std::mt19937&)>;
	using Changer =  std::function<void(Population&, std::mt19937&)>;

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
	Changer mate;
	Changer mut;

	public:

	BaseGenetic(Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Changer mate, Changer mut):
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

	BaseGenetic(int max_gen, int pop_size, int elite, Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Changer mate, Changer mut):
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

	private:

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
	using Changer =  std::function<void(Population&, std::mt19937&)>;

	Population population;

	public:

	bool output;

	SimpleGenetic(Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Changer mate, Changer mut):
		BaseGenetic<GeneType, FitnessType>(gen, comp, eval, stop, sel, mate, mut),
		population(this->pop_size),
		output(true)
	{	
	}

	SimpleGenetic(int max_gen, int pop_size, int elite, Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Changer mate, Changer mut):
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
		this->mate(pop, this->engine);
		this->mut(pop, this->engine);

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
class AdvancedGeneticAlgorithm: BaseGenetic<GeneType, FitnessType>
{	
	using Individual = BaseIndividual<GeneType, FitnessType>;
	using Population = std::vector<Individual>;
	using Generator = std::function<Individual(void)>;
	using Comparator = std::function<bool(const Individual&, const Individual&)>;
	using Evaluate = std::function<FitnessType(Individual&)>;
	using StopCond = std::function<bool(Individual&)>;
	using Selection = std::function<void(Population&, Comparator, std::mt19937&)>;
	using Changer =  std::function<void(Population&, std::mt19937&)>;

	Population population;

	public:

	AdvancedGeneticAlgorithm(Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Changer mate, Changer mut):
		BaseGenetic<GeneType, FitnessType>(gen, comp, eval, stop, sel, mate, mut),
		population(this->pop_size),
	{	}

	AdvancedGeneticAlgorithm(int max_gen, int pop_size, int elite, Generator gen, Comparator comp, Evaluate eval, StopCond stop, Selection sel, Changer mate, Changer mut):
		BaseGenetic<GeneType, FitnessType>(max_gen, pop_size, elite, gen, comp, eval, stop, sel, mate, mut),
		population(this->pop_size),
	{	}

	Individual operator()(void)
	{
		int threads_count = std::thread::hardware_concurrency();
		thread_pool pool(threads_count);


	}

	private:

	int compute_fitness()
	{
		int evaluated_counter = 0;

		for(auto i = pop.begin(); i != pop.end(); i++)
		{
			if(!i->is_valid())
			{
				i->compute_fitness(this->evaluate);
				evaluated_counter++;
			}
		}

		return evaluated_counter;
	}

	void one_iter(Population &pop)
	{
		this->sel(pop, this->comparator, this->engine);
		this->mate(pop, this->engine);
		this->mut(pop, this->engine);
	}
};

} // namespace ga

#endif
